# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from typing import Any, Dict, List, Optional, Sequence

from opentelemetry._events import Event
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.trace import ReadableSpan

UNKNOWN_INSTRUMENTATION_SCOPE = "unknown"

# Traceloop constants
TRACELOOP_INSTRUMENTATION_SCOPE_PREFIX = "opentelemetry.instrumentation"
TRACELOOP_ATTRIBUTE_PREFIX = "traceloop"
TRACELOOP_ENTITY_INPUT = "traceloop.entity.input"
TRACELOOP_ENTITY_OUTPUT = "traceloop.entity.output"
TRACELOOP_CREW_TASKS_OUTPUT = "crewai.crew.tasks_output"
TRACELOOP_CREW_RESULT = "crewai.crew.result"

# OpenInference constants
OPENINFERENCE_INSTRUMENTATION_SCOPE_PREFIX = "openinference.instrumentation"
OPENINFERENCE_INPUT_VALUE = "input.value"
OPENINFERENCE_OUTPUT_VALUE = "output.value"

# Roles
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


class LLOHandler:
    """
    Utility class for handling Large Language Objects (LLO) in OpenTelemetry spans.

    LLOHandler performs three primary functions:
    1. Identifies input/output prompt content in spans
    2. Extracts and transforms these attributes into an OpenTelemetry Gen AI Event
    3. Filters input/output prompts from spans to maintain privacy and reduce span size

    This LLOHandler supports the following third-party instrumentation libraries:
    - Strands
    - OpenInference
    - Traceloop/OpenLLMetry
    - OpenLIT
    """

    def __init__(self, logger_provider: LoggerProvider):
        """
        Initialize an LLOHandler with the specified logger provider.

        This constructor sets up the event logger provider, configures the event logger,
        and initializes the patterns used to identify LLO attributes.

        Args:
            logger_provider: The OpenTelemetry LoggerProvider used for emitting events.
                           Global LoggerProvider instance injected from our AwsOpenTelemetryConfigurator
        """
        self._logger_provider = logger_provider

        self._event_logger_provider = EventLoggerProvider(logger_provider=self._logger_provider)

        # Patterns for attribute filtering - using a set for O(1) lookups
        self._traceloop_exact_match_patterns = {
            TRACELOOP_ENTITY_INPUT,
            TRACELOOP_ENTITY_OUTPUT,
            TRACELOOP_CREW_TASKS_OUTPUT,
            TRACELOOP_CREW_RESULT,
        }

        self._regex_patterns = [
            re.compile(r"^gen_ai\.prompt\.\d+\.content$"),
            re.compile(r"^gen_ai\.completion\.\d+\.content$"),
        ]


    def process_spans(self, spans: Sequence[ReadableSpan]) -> List[ReadableSpan]:
        modified_spans = []

        for span in spans:
            self._emit_llo_attributes(span, span.attributes)
            updated_attributes = self._filter_attributes(span.attributes)

            if isinstance(span.attributes, BoundedAttributes):
                span._attributes = BoundedAttributes(
                    maxlen=span.attributes.maxlen,
                    attributes=updated_attributes,
                    immutable=span.attributes._immutable,
                    max_value_len=span.attributes.max_value_len,
                )
            else:
                span._attributes = updated_attributes

            self._process_span_events(span)

            modified_spans.append(span)

        return modified_spans


    def _process_span_events(self, span: ReadableSpan) -> None:
        pass


    def _emit_llo_attributes(
        self, span: ReadableSpan, attributes: Dict[str, Any], event_timestamp: Optional[int] = None
    ):
        # Collect all messages from various third-party instrumentations
        messages = []

        if self._is_traceloop_span(span):
            messages.extend(self._collect_traceloop_messages(attributes))
        elif self._is_openinference_span(span):
            pass

        input_messages = []
        output_messages = []

        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")

            # Determine if message is input or output based on role and context
            if role in [ROLE_SYSTEM, ROLE_USER]:
                input_messages.append({
                    "role": role,
                    "content": content
                })
            elif role == ROLE_ASSISTANT:
                output_messages.append({
                    "role": role,
                    "content": content
                })
            else:
                # For unknown roles, try to determine based on context
                # If it's from completion/output attribute, treat as output
                if any(key in message.get("source", "") for key in ["completion", "output", "result"]):
                    output_messages.append({
                        "role": role,
                        "content": content
                    })
                else:
                    input_messages.append({
                        "role": role,
                        "content": content
                    })

        event_body = {}
        if input_messages:
            event_body["input"] = {
                "messages": input_messages
            }
        if output_messages:
            event_body["output"] = {
                "messages": output_messages
            }

        # Create a single consolidated event: one span -> one event
        span_ctx = span.context

        # Copy scope name from span
        scope_name = span.instrumentation_scope.name if span.instrumentation_scope else UNKNOWN_INSTRUMENTATION_SCOPE

        event_logger = self._event_logger_provider.get_event_logger(scope_name)

        event_logger.emit(Event(
            name=span.instrumentation_scope.name if span.instrumentation_scope else UNKNOWN_INSTRUMENTATION_SCOPE,
            timestamp = event_timestamp if event_timestamp is not None else span.end_time,
            body=event_body,
            trace_id=span_ctx.trace_id if span_ctx else None,
            span_id=span_ctx.span_id if span_ctx else None,
            trace_flags=span_ctx.trace_flags if span_ctx else None,
        ))


    def _is_traceloop_span(self, span: ReadableSpan) -> bool:
        return (
            span.instrumentation_scope is not None and
            span.instrumentation_scope.name.startswith(TRACELOOP_INSTRUMENTATION_SCOPE_PREFIX) and
            span.attributes is not None and
            any(TRACELOOP_ATTRIBUTE_PREFIX in key for key in span.attributes.keys())
        )


    def _is_openinference_span(self, span: ReadableSpan) -> bool:
        return (
            span.instrumentation_scope is not None and
            span.instrumentation_scope.name.startswith(OPENINFERENCE_INSTRUMENTATION_SCOPE_PREFIX)
        )


    def _collect_traceloop_messages(self, attributes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract Traceloop input/output messages from span attributes."""
        traceloop_messages = []

        traceloop_attrs = [
            (TRACELOOP_ENTITY_INPUT, ROLE_USER, "input"),
            (TRACELOOP_ENTITY_OUTPUT, ROLE_ASSISTANT, "output"),
            (TRACELOOP_CREW_TASKS_OUTPUT, ROLE_ASSISTANT, "output"),
            (TRACELOOP_CREW_RESULT, ROLE_ASSISTANT, "output"),
        ]

        traceloop_input_msg_pattern = re.compile(r"^gen_ai\.prompt\.(\d+)\.content$")
        traceloop_output_msg_pattern = re.compile(r"^gen_ai\.completion\.(\d+)\.content$")

        if not any(attr_key in attributes for attr_key, _, _ in traceloop_attrs):
            return []

        for attr_key, role, source in traceloop_attrs:
            if attr_key in attributes:
                traceloop_messages.append(
                    {
                        "content": attributes[attr_key],
                        "role": role,
                        "source": source
                    }
                )

        input_messages = {}
        for key, value in attributes.items():
            match = traceloop_input_msg_pattern.match(key)
            if match:
                index = match.group(1)
                role_key = f"gen_ai.prompt.{index}.role"
                role = attributes.get(role_key, "unknown")
                input_messages[index] = (value, role)

        for index in sorted(input_messages.keys(), key=int):
            value, role = input_messages[index]
            traceloop_messages.append({
                "content": value,
                "role": role,
                "source": "input"
            })

        output_messages = {}
        for key, value in attributes.items():
            match = traceloop_output_msg_pattern.match(key)
            if match:
                index = match.group(1)
                role_key = f"gen_ai.completion.{index}.role"
                role = attributes.get(role_key, "unknown")
                output_messages[index] = (value, role)

        for index in sorted(output_messages.keys(), key=int):
            value, role = input_messages[index]
            traceloop_messages.append({
                "content": value,
                "role": role,
                "source": "output"
            })

        return traceloop_messages


    def _collect_openinference_messages(self, attributes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract OpenInference input/output messages from span attributes."""
        openinference_messages = []

        openinference_attrs = [
            (OPENINFERENCE_INPUT_VALUE, ROLE_USER, "input"),
            (OPENINFERENCE_OUTPUT_VALUE, ROLE_ASSISTANT, "output"),
        ]

        openinference_input_msg_pattern = re.compile(r"^llm\.input_messages\.(\d+)\.message\.content$")
        openinference_output_msg_pattern = re.compile(r"^llm\.output_messages\.(\d+)\.message\.content$")

        for attr_key, role, source in openinference_attrs:
            if attr_key in attributes:
                openinference_messages.append({
                    "content": attributes[attr_key],
                    "role": role,
                    "source": source
                })

        input_messages = {}
        for key, value in attributes.items():
            match = openinference_input_msg_pattern.match(key)
            if match:
                index = match.group(1)
                role_key = f"llm.input_messages.{index}.message.role"
                role = attributes.get(role_key, ROLE_USER)
                input_messages[index] = (value, role)

        for index in sorted(input_messages.keys(), key=int):
            value, role = input_messages[index]
            openinference_messages.append({
                "content": value,
                "role": role,
                "source": "input"
            })

        output_messages = {}
        for key, value in attributes.items():
            match = openinference_output_msg_pattern.match(key)
            if match:
                index = match.group(1)
                role_key = f"llm.output_messages.{index}.message.role"
                role = attributes.get(role_key, ROLE_ASSISTANT)
                output_messages[index] = (value, role)

        for index in sorted(output_messages.keys(), key=int):
            value, role = input_messages[index]
            openinference_messages.append({
                "content": value,
                "role": role,
                "source": "output"
            })

        return openinference_messages
