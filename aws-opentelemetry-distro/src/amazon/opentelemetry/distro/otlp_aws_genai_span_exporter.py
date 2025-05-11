import requests

from typing import Dict, Optional, Sequence
from amazon.opentelemetry.distro.llo_handler import LLOHandler
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

class OTLPAwsGenAiSpanExporter(OTLPSpanExporter):
    def __init__(
        self,
        logs_exporter: OTLPLogExporter,
        endpoint: Optional[str] = None,
        certificate_file: Optional[str] = None,
        client_key_file: Optional[str] = None,
        client_certificate_file: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        compression: Optional[Compression] = None,
        session: Optional[requests.Session] = None,
    ):
        self._llo_handler = LLOHandler(logs_exporter)

        super().__init__(
            endpoint=endpoint,
            certificate_file=certificate_file,
            client_key_file=client_key_file,
            client_certificate_file=client_certificate_file,
            headers=headers,
            timeout=timeout,
            compression=compression,
            session=session
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        spans_to_export = self._llo_handler.process_spans(spans)

        return super().export(spans_to_export)
