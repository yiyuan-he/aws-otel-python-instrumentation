# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from typing import Dict, Optional, Sequence

import re
import requests

from amazon.opentelemetry.distro._utils import is_installed
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

AWS_SERVICE = "xray"
AWS_CLOUDWATCH_LOG_GROUP_ENV = "AWS_CLOUDWATCH_LOG_GROUP"
AWS_CLOUDWATCH_LOG_STREAM_ENV = "AWS_CLOUDWATCH_LOG_STREAM"
_logger = logging.getLogger(__name__)

class LLOSenderClient:
    """Skeleton client for handling Large Language Objects (LLO)"""

    def __init__(self, bucket_name: str = "mock-bucket", region_name: Optional[str] = None):
        self._bucket_name = bucket_name
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"Initialized mock LLO sender client with bucket: {bucket_name}")

    def upload(self, data: str, metadata: Dict[str, str]) -> str:
        """Mock upload that returns a dummy S3 pointer"""
        attribute_name = metadata.get("attribute_name", "unknown")
        self._logger.debug(f"Mock upload of LLO attribute: {attribute_name}")
        return f"s3://{self._bucket_name}/{metadata.get('trace_id', 'trace')}/{metadata.get('span_id', 'span')}/{attribute_name}"

    def shutdown(self):
        """Mock shutdown"""
        self._logger.debug("Mock LLO sender client shutdown")


class OTLPAwsSpanExporter(OTLPSpanExporter):
    """
    This exporter extends the functionality of the OTLPSpanExporter to allow spans to be exported to the
    XRay OTLP endpoint https://xray.[AWSRegion].amazonaws.com/v1/traces. Utilizes the botocore
    library to sign and directly inject SigV4 Authentication to the exported request's headers.

    https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch-OTLPEndpoint.html
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        certificate_file: Optional[str] = None,
        client_key_file: Optional[str] = None,
        client_certificate_file: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        compression: Optional[Compression] = None,
        rsession: Optional[requests.Session] = None,
    ):

        self._aws_region = None
        self._has_required_dependencies = False
        # Requires botocore to be installed to sign the headers. However,
        # some users might not need to use this exporter. In order not conflict
        # with existing behavior, we check for botocore before initializing this exporter.

        if endpoint and is_installed("botocore"):
            # pylint: disable=import-outside-toplevel
            from botocore import auth, awsrequest, session

            self.boto_auth = auth
            self.boto_aws_request = awsrequest
            self.boto_session = session.Session()

            # Assumes only valid endpoints passed are of XRay OTLP format.
            # The only usecase for this class would be for ADOT Python Auto Instrumentation and that already validates
            # the endpoint to be an XRay OTLP endpoint.
            self._aws_region = endpoint.split(".")[1]
            self._has_required_dependencies = True

        else:
            _logger.error(
                "botocore is required to export traces to %s. Please install it using `pip install botocore`",
                endpoint,
            )

        self._llo_sender = LLOSenderClient(bucket_name="my-telemetry-bucket", region_name=self._aws_region)

        super().__init__(
            endpoint=endpoint,
            certificate_file=certificate_file,
            client_key_file=client_key_file,
            client_certificate_file=client_certificate_file,
            headers=headers,
            timeout=timeout,
            compression=compression,
            session=rsession,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        modified_spans = []

        for span in spans:
            # Create updated attributes
            update_attributes = {}

            # Copy all original attributes and handle LLO data
            for key, value in span.attributes.items():
                if self._should_offload(key, value):
                    metadata = {
                        "trace_id": format(span.context.trace_id, 'x'),
                        "span_id": format(span.context.span_id, 'x'),
                        "attribute_name": key,
                        "span_name": span.name
                    }

                    # Get pointer from LLO client
                    # Q: How to handle if LLO client operation fails?
                    try:
                        pointer = self._llo_sender.upload(value, metadata)

                        # Store pointer instead of original value
                        update_attributes[key] = pointer
                        _logger.debug(f"Offloaded LLO attribute {key}")
                    except Exception as e:
                        # If offloading fails, keep original value
                        _logger.warning(f"Failed to offload LLO attribute {key}: {e}")
                        update_attributes[key] = value
                else:
                    # Keep original value
                    update_attributes[key] = value

            # Create a new span with updated attributes
            if isinstance(span.attributes, BoundedAttributes):
                span._attributes = BoundedAttributes(
                    maxlen=span.attributes.maxlen,
                    attributes=update_attributes,
                    immutable=span.attributes._immutable,
                    max_value_len=span.attributes.max_value_len
                )
            else:
                span._attributes = update_attributes

            modified_spans.append(span)

        # Call the parent's export method
        return super().export(modified_spans)

    def _should_offload(self, key, value):
        """Determine if attribute should be offloaded to S3"""
        if not isinstance(value, str):
            return False

        # Exact match LLO attributes
        exact_match_patterns = [
            # Exact match patterns
            "traceloop.entity.input", 
            "traceloop.entity.output", 
        ]

        # Regex match LLO attributes
        regex_patterns = [
            # Regex patterns
            r"^gen_ai\.prompt\.\d+\.content$",
            r"^gen_ai\.completion\.\d+\.content$"
        ]

        # Check if attribute matches patterns for offloading
        return (
            any(pattern in key for pattern in exact_match_patterns) or
            any(re.match(pattern, key) for pattern in regex_patterns)
        )

    # Overrides upstream's private implementation of _export. All behaviors are
    # the same except if the endpoint is an XRay OTLP endpoint, we will sign the request
    # with SigV4 in headers before sending it to the endpoint. Otherwise, we will skip signing.
    def _export(self, serialized_data: bytes):
        if self._has_required_dependencies:
            request = self.boto_aws_request.AWSRequest(
                method="POST",
                url=self._endpoint,
                data=serialized_data,
                headers={"Content-Type": "application/x-protobuf"},
            )

            # Add CloudWatch Log Group and Log Stream headers if configured
            cloudwatch_log_group = os.environ.get(AWS_CLOUDWATCH_LOG_GROUP_ENV)
            cloudwatch_log_stream = os.environ.get(AWS_CLOUDWATCH_LOG_STREAM_ENV)

            if cloudwatch_log_group:
                request.headers["x-aws-log-group"] = cloudwatch_log_group
                _logger.debug("Adding CloudWatch Log Group header: %s", cloudwatch_log_group)

            if cloudwatch_log_stream:
                request.headers["x-aws-log-stream"] = cloudwatch_log_stream
                _logger.debug("Adding CloudWatch Log Stream header: %s", cloudwatch_log_stream)

            credentials = self.boto_session.get_credentials()

            if credentials is not None:
                signer = self.boto_auth.SigV4Auth(credentials, AWS_SERVICE, self._aws_region)

                try:
                    signer.add_auth(request)
                    self._session.headers.update(dict(request.headers))

                except Exception as signing_error:  # pylint: disable=broad-except
                    _logger.error("Failed to sign request: %s", signing_error)
        else:
            _logger.debug("botocore is not installed. Failed to sign request to export traces to: %s", self._endpoint)

        return super()._export(serialized_data)
