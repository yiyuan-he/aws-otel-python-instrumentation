import logging
import os
from typing import Dict, Optional, Sequence

import requests

from amazon.opentelemetry.distro._utils import is_installed
from opentelemetry.exporter.otlp.proto.http import Compression
try:
    # Try to import from new path in newer versions
    from opentelemetry.exporter.otlp.proto.http.logs_exporter import OTLPLogExporter
except ImportError:
    # Fall back to older import path in older versions
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LogRecord
from opentelemetry.sdk._logs.export import LogExportResult

# For CloudWatch Logs, the service name is 'logs' not 'xray'
AWS_SERVICE = "logs"
AWS_CLOUDWATCH_LOG_GROUP_ENV = "AWS_CLOUDWATCH_LOG_GROUP"
AWS_CLOUDWATCH_LOG_STREAM_ENV = "AWS_CLOUDWATCH_LOG_STREAM"
# Set up more verbose logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
# Add a console handler if not already present
if not _logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)
_logger.debug("OTLPAwsLogExporter module loaded")

class OTLPAwsLogExporter(OTLPLogExporter):
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
        
        _logger.info(f"Initializing OTLPAwsLogExporter with endpoint: {endpoint}")

        if endpoint and is_installed("botocore"):
            # pylint: disable=import-outside-toplevel
            from botocore import auth, awsrequest, session
            self.boto_auth = auth
            self.boto_aws_request = awsrequest
            self.boto_session = session.Session()

            # For logs endpoint https://logs.[region].amazonaws.com/v1/logs
            self._aws_region = endpoint.split(".")[1]
            self._has_required_dependencies = True
            _logger.info(f"Successfully configured SigV4 authentication for region: {self._aws_region}")
        else:
            _logger.error(
                "botocore is required to export logs to %s. Please install it using `pip install botocore`",
                endpoint,
            )

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

    def _export(self, serialized_data: bytes):
        _logger.debug(f"Exporting logs to {self._endpoint}")
        try:
            if self._has_required_dependencies:
                request = self.boto_aws_request.AWSRequest(
                    method="POST",
                    url=self._endpoint,
                    data=serialized_data,
                    headers={"Content-Type": "application/x-protobuf"},
                )

                cloudwatch_log_group = os.environ.get(AWS_CLOUDWATCH_LOG_GROUP_ENV)
                cloudwatch_log_stream = os.environ.get(AWS_CLOUDWATCH_LOG_STREAM_ENV)

                if cloudwatch_log_group:
                    request.headers["x-aws-log-group"] = cloudwatch_log_group
                    _logger.debug(f"Adding CloudWatch Log Group header: {cloudwatch_log_group}")

                if cloudwatch_log_stream:
                    request.headers["x-aws-log-stream"] = cloudwatch_log_stream
                    _logger.debug(f"Adding CloudWatch Log Stream header: {cloudwatch_log_stream}")

                credentials = self.boto_session.get_credentials()

                if credentials is not None:
                    _logger.debug(f"Signing request with SigV4 for service: {AWS_SERVICE}, region: {self._aws_region}")
                    signer = self.boto_auth.SigV4Auth(credentials, AWS_SERVICE, self._aws_region)

                    try:
                        signer.add_auth(request)
                        self._session.headers.update(dict(request.headers))
                        _logger.debug("Request signed successfully")
                    except Exception as signing_error: # pylint: disable=broad-except
                        _logger.error(f"Failed to sign request: {signing_error}")
                else:
                    _logger.error("Failed to obtain AWS credentials for SigV4 signing")
            else:
                _logger.warning(f"SigV4 authentication not available for {self._endpoint}. Falling back to unsigned request.")

            _logger.debug("Calling parent _export method")
            result = super()._export(serialized_data)
            _logger.debug(f"Parent _export method returned: {result}")
            return result
        except Exception as e:
            _logger.exception(f"Exception in _export: {str(e)}")
            # Still try to call the parent method in case it can handle the error
            return super()._export(serialized_data)
