# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest import TestCase
from unittest.mock import MagicMock, patch

from amazon.opentelemetry.distro.llo_handler import LLOHandler
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.trace import Event as SpanEvent
from opentelemetry.sdk.trace import ReadableSpan, SpanContext
from opentelemetry.trace import SpanKind, TraceFlags, TraceState


class TestLLOHandler(TestCase):
    def setUp(self):
        self.logger_provider_mock = MagicMock(spec=LoggerProvider)
        self.event_logger_mock = MagicMock()
        self.event_logger_provider_mock = MagicMock()
        self.event_logger_provider_mock.get_event_logger.return_value = self.event_logger_mock

        with patch(
            "amazon.opentelemetry.distro.llo_handler.EventLoggerProvider", return_value=self.event_logger_provider_mock
        ):
            self.llo_handler = LLOHandler(self.logger_provider_mock)

    def _create_mock_span(self, attributes=None, kind=SpanKind.INTERNAL):
        """
        Helper method to create a mock span with given attributes
        """
        if attributes is None:
            attributes = {}

        span_context = SpanContext(
            trace_id=0x123456789ABCDEF0123456789ABCDEF0,
            span_id=0x123456789ABCDEF0,
            is_remote=False,
            trace_flags=TraceFlags.SAMPLED,
            trace_state=TraceState.get_default(),
        )

        mock_span = MagicMock(spec=ReadableSpan)
        mock_span.context = span_context
        mock_span.attributes = attributes
        mock_span.kind = kind
        mock_span.start_time = 1234567890

        return mock_span

    def test_init(self):
        """
        Test initialization of LLOHandler
        """
        self.assertEqual(self.llo_handler._logger_provider, self.logger_provider_mock)
        self.assertEqual(self.llo_handler._event_logger_provider, self.event_logger_provider_mock)

    def test_is_llo_attribute_match(self):
        """
        Test _is_llo_attribute method with matching patterns
        """
        self.assertTrue(self.llo_handler._is_llo_attribute("gen_ai.prompt.0.content"))
        self.assertTrue(self.llo_handler._is_llo_attribute("gen_ai.prompt.123.content"))

    def test_is_llo_attribute_no_match(self):
        """
        Test _is_llo_attribute method with non-matching patterns
        """
        self.assertFalse(self.llo_handler._is_llo_attribute("gen_ai.prompt.content"))
        self.assertFalse(self.llo_handler._is_llo_attribute("gen_ai.prompt.abc.content"))
        self.assertFalse(self.llo_handler._is_llo_attribute("some.other.attribute"))

    def test_is_llo_attribute_traceloop_match(self):
        """
        Test _is_llo_attribute method with Traceloop patterns
        """
        # Test exact matches for Traceloop attributes
        self.assertTrue(self.llo_handler._is_llo_attribute("traceloop.entity.input"))
        self.assertTrue(self.llo_handler._is_llo_attribute("traceloop.entity.output"))

    def test_is_llo_attribute_openlit_match(self):
        """
        Test _is_llo_attribute method with OpenLit patterns
        """
        # Test exact matches for direct OpenLit attributes
        self.assertTrue(self.llo_handler._is_llo_attribute("gen_ai.prompt"))
        self.assertTrue(self.llo_handler._is_llo_attribute("gen_ai.completion"))
        self.assertTrue(self.llo_handler._is_llo_attribute("gen_ai.content.revised_prompt"))

    def test_is_llo_attribute_openinference_match(self):
        """
        Test _is_llo_attribute method with OpenInference patterns
        """
        # Test exact matches
        self.assertTrue(self.llo_handler._is_llo_attribute("input.value"))
        self.assertTrue(self.llo_handler._is_llo_attribute("output.value"))

        # Test regex matches
        self.assertTrue(self.llo_handler._is_llo_attribute("llm.input_messages.0.message.content"))
        self.assertTrue(self.llo_handler._is_llo_attribute("llm.output_messages.123.message.content"))

    def test_is_llo_attribute_crewai_match(self):
        """
        Test _is_llo_attribute method with CrewAI patterns
        """
        # Test exact match for CrewAI attributes (handled by Traceloop and OpenLit)
        self.assertTrue(self.llo_handler._is_llo_attribute("gen_ai.agent.actual_output"))
        self.assertTrue(self.llo_handler._is_llo_attribute("gen_ai.agent.human_input"))
        self.assertTrue(self.llo_handler._is_llo_attribute("crewai.crew.tasks_output"))
        self.assertTrue(self.llo_handler._is_llo_attribute("crewai.crew.result"))

    def test_filter_attributes(self):
        """
        Test _filter_attributes method
        """
        attributes = {
            "gen_ai.prompt.0.content": "test content",
            "gen_ai.prompt.0.role": "user",
            "normal.attribute": "value",
            "another.normal.attribute": 123,
        }

        filtered = self.llo_handler._filter_attributes(attributes)

        self.assertNotIn("gen_ai.prompt.0.content", filtered)
        self.assertIn("gen_ai.prompt.0.role", filtered)
        self.assertIn("normal.attribute", filtered)
        self.assertIn("another.normal.attribute", filtered)

    def test_collect_gen_ai_prompt_messages_system_role(self):
        """
        Test _collect_gen_ai_prompt_messages with system role
        """
        attributes = {
            "gen_ai.prompt.0.content": "system instruction",
            "gen_ai.prompt.0.role": "system",
            "gen_ai.system": "openai",
        }

        span = self._create_mock_span(attributes)

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["content"], "system instruction")
        self.assertEqual(message["role"], "system")
        self.assertEqual(message["source"], "prompt")

    def test_collect_gen_ai_prompt_messages_user_role(self):
        """
        Test _collect_gen_ai_prompt_messages with user role
        """
        attributes = {
            "gen_ai.prompt.0.content": "user question",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.system": "anthropic",
        }

        span = self._create_mock_span(attributes)

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["content"], "user question")
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["source"], "prompt")

    def test_collect_gen_ai_prompt_messages_assistant_role(self):
        """
        Test _collect_gen_ai_prompt_messages with assistant role
        """
        attributes = {
            "gen_ai.prompt.1.content": "assistant response",
            "gen_ai.prompt.1.role": "assistant",
            "gen_ai.system": "anthropic",
        }

        span = self._create_mock_span(attributes)

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["content"], "assistant response")
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["source"], "prompt")

    def test_collect_gen_ai_prompt_messages_function_role(self):
        """
        Test _collect_gen_ai_prompt_messages with function role
        """
        attributes = {
            "gen_ai.prompt.2.content": "function data",
            "gen_ai.prompt.2.role": "function",
            "gen_ai.system": "openai",
        }

        span = self._create_mock_span(attributes)
        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["content"], "function data")
        self.assertEqual(message["role"], "function")
        self.assertEqual(message["source"], "prompt")

    def test_collect_gen_ai_prompt_messages_unknown_role(self):
        """
        Test _collect_gen_ai_prompt_messages with unknown role
        """
        attributes = {
            "gen_ai.prompt.3.content": "unknown type content",
            "gen_ai.prompt.3.role": "unknown",
            "gen_ai.system": "bedrock",
        }

        span = self._create_mock_span(attributes)
        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["content"], "unknown type content")
        self.assertEqual(message["role"], "unknown")
        self.assertEqual(message["source"], "prompt")

    def test_collect_gen_ai_completion_messages_assistant_role(self):
        """
        Test _collect_gen_ai_completion_messages with assistant role
        """
        attributes = {
            "gen_ai.completion.0.content": "assistant completion",
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.system": "openai",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899  # end time for completion events

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["content"], "assistant completion")
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["source"], "completion")

    def test_collect_gen_ai_completion_messages_other_role(self):
        """
        Test _collect_gen_ai_completion_messages with non-assistant role
        """
        attributes = {
            "gen_ai.completion.1.content": "other completion",
            "gen_ai.completion.1.role": "other",
            "gen_ai.system": "anthropic",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["content"], "other completion")
        self.assertEqual(message["role"], "other")
        self.assertEqual(message["source"], "completion")

    def test_collect_traceloop_messages(self):
        """
        Test collection of standard Traceloop attributes
        """
        attributes = {
            "traceloop.entity.input": "input data",
            "traceloop.entity.output": "output data",
            "traceloop.entity.name": "my_entity",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        # Filter to only Traceloop messages
        traceloop_messages = [m for m in messages if m["source"] in ["input", "output"]]

        self.assertEqual(len(traceloop_messages), 2)

        input_message = traceloop_messages[0]
        self.assertEqual(input_message["content"], "input data")
        self.assertEqual(input_message["role"], "user")
        self.assertEqual(input_message["source"], "input")

        output_message = traceloop_messages[1]
        self.assertEqual(output_message["content"], "output data")
        self.assertEqual(output_message["role"], "assistant")
        self.assertEqual(output_message["source"], "output")

    def test_collect_traceloop_messages_all_attributes(self):
        """
        Test _collect_traceloop_messages with all Traceloop attributes including CrewAI outputs
        """
        attributes = {
            "traceloop.entity.input": "input data",
            "traceloop.entity.output": "output data",
            "crewai.crew.tasks_output": "[TaskOutput(description='Task 1', output='Result 1')]",
            "crewai.crew.result": "Final crew result",
            "traceloop.entity.name": "crewai_agent",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 4)

        # Check standard Traceloop messages
        self.assertEqual(messages[0]["content"], "input data")
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["source"], "input")

        self.assertEqual(messages[1]["content"], "output data")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["source"], "output")

        # Check CrewAI messages
        self.assertEqual(messages[2]["content"], "[TaskOutput(description='Task 1', output='Result 1')]")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["source"], "output")

        self.assertEqual(messages[3]["content"], "Final crew result")
        self.assertEqual(messages[3]["role"], "assistant")
        self.assertEqual(messages[3]["source"], "result")

    def test_collect_openlit_messages_direct_prompt(self):
        """
        Test _collect_openlit_messages with direct prompt attribute
        """
        attributes = {"gen_ai.prompt": "user direct prompt", "gen_ai.system": "openlit"}

        span = self._create_mock_span(attributes)

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["content"], "user direct prompt")
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["source"], "prompt")

    def test_collect_openlit_messages_direct_completion(self):
        """
        Test _collect_openlit_messages with direct completion attribute
        """
        attributes = {"gen_ai.completion": "assistant direct completion", "gen_ai.system": "openlit"}

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["content"], "assistant direct completion")
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["source"], "completion")

    def test_collect_openlit_messages_all_attributes(self):
        """
        Test _collect_openlit_messages with all OpenLit attributes
        """
        attributes = {
            "gen_ai.prompt": "user prompt",
            "gen_ai.completion": "assistant response",
            "gen_ai.content.revised_prompt": "revised prompt",
            "gen_ai.agent.actual_output": "agent output",
            "gen_ai.agent.human_input": "human input to agent",
            "gen_ai.system": "langchain",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 5)

        # Check message contents and roles
        self.assertEqual(messages[0]["content"], "user prompt")
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["source"], "prompt")

        self.assertEqual(messages[1]["content"], "assistant response")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["source"], "completion")

        self.assertEqual(messages[2]["content"], "revised prompt")
        self.assertEqual(messages[2]["role"], "system")
        self.assertEqual(messages[2]["source"], "prompt")

        self.assertEqual(messages[3]["content"], "agent output")
        self.assertEqual(messages[3]["role"], "assistant")
        self.assertEqual(messages[3]["source"], "output")

        self.assertEqual(messages[4]["content"], "human input to agent")
        self.assertEqual(messages[4]["role"], "user")
        self.assertEqual(messages[4]["source"], "input")

    def test_collect_openlit_messages_revised_prompt(self):
        """
        Test _collect_openlit_messages with revised prompt attribute
        """
        attributes = {"gen_ai.content.revised_prompt": "revised system prompt", "gen_ai.system": "openlit"}

        span = self._create_mock_span(attributes)

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["content"], "revised system prompt")
        self.assertEqual(message["role"], "system")
        self.assertEqual(message["source"], "prompt")

    def test_collect_openinference_messages_direct_attributes(self):
        """
        Test _collect_openinference_messages with direct input/output values
        """
        attributes = {
            "input.value": "user prompt",
            "output.value": "assistant response",
            "llm.model_name": "gpt-4",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 2)

        input_message = messages[0]
        self.assertEqual(input_message["content"], "user prompt")
        self.assertEqual(input_message["role"], "user")
        self.assertEqual(input_message["source"], "input")

        output_message = messages[1]
        self.assertEqual(output_message["content"], "assistant response")
        self.assertEqual(output_message["role"], "assistant")
        self.assertEqual(output_message["source"], "output")

    def test_collect_openinference_messages_structured_input(self):
        """
        Test _collect_openinference_messages with structured input messages
        """
        attributes = {
            "llm.input_messages.0.message.content": "system prompt",
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.1.message.content": "user message",
            "llm.input_messages.1.message.role": "user",
            "llm.model_name": "claude-3",
        }

        span = self._create_mock_span(attributes)

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 2)

        system_message = messages[0]
        self.assertEqual(system_message["content"], "system prompt")
        self.assertEqual(system_message["role"], "system")
        self.assertEqual(system_message["source"], "input")

        user_message = messages[1]
        self.assertEqual(user_message["content"], "user message")
        self.assertEqual(user_message["role"], "user")
        self.assertEqual(user_message["source"], "input")

    def test_collect_openinference_messages_structured_output(self):
        """
        Test _collect_openinference_messages with structured output messages
        """
        attributes = {
            "llm.output_messages.0.message.content": "assistant response",
            "llm.output_messages.0.message.role": "assistant",
            "llm.model_name": "llama-3",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 1)

        output_message = messages[0]
        self.assertEqual(output_message["content"], "assistant response")
        self.assertEqual(output_message["role"], "assistant")
        self.assertEqual(output_message["source"], "output")

    def test_collect_openinference_messages_mixed_attributes(self):
        """
        Test _collect_openinference_messages with a mix of all attribute types
        """
        attributes = {
            "input.value": "direct input",
            "output.value": "direct output",
            "llm.input_messages.0.message.content": "message input",
            "llm.input_messages.0.message.role": "user",
            "llm.output_messages.0.message.content": "message output",
            "llm.output_messages.0.message.role": "assistant",
            "llm.model_name": "bedrock.claude-3",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 4)

        # Check that we got all the expected messages
        contents = [msg["content"] for msg in messages]
        self.assertIn("direct input", contents)
        self.assertIn("direct output", contents)
        self.assertIn("message input", contents)
        self.assertIn("message output", contents)

        # Check roles
        roles = [msg["role"] for msg in messages]
        self.assertEqual(roles.count("user"), 2)
        self.assertEqual(roles.count("assistant"), 2)

    def test_collect_openlit_messages_agent_actual_output(self):
        """
        Test _collect_openlit_messages with agent actual output attribute
        """
        attributes = {"gen_ai.agent.actual_output": "Agent task output result", "gen_ai.system": "crewai"}

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 1)

        message = messages[0]
        self.assertEqual(message["content"], "Agent task output result")
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["source"], "output")

    def test_collect_openlit_messages_agent_human_input(self):
        """
        Test _collect_openlit_messages with agent human input attribute
        """
        attributes = {"gen_ai.agent.human_input": "Human input to the agent", "gen_ai.system": "crewai"}

        span = self._create_mock_span(attributes)

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["content"], "Human input to the agent")
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["source"], "input")

    def test_collect_traceloop_messages_crew_outputs(self):
        """
        Test _collect_traceloop_messages with CrewAI specific attributes
        """
        attributes = {
            "crewai.crew.tasks_output": "[TaskOutput(description='Task description', output='Task result')]",
            "crewai.crew.result": "Final crew execution result",
            "traceloop.entity.name": "crewai",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 2)

        # Check the tasks output message
        tasks_message = messages[0]
        self.assertEqual(tasks_message["content"], "[TaskOutput(description='Task description', output='Task result')]")
        self.assertEqual(tasks_message["role"], "assistant")
        self.assertEqual(tasks_message["source"], "output")

        # Check the result message
        result_message = messages[1]
        self.assertEqual(result_message["content"], "Final crew execution result")
        self.assertEqual(result_message["role"], "assistant")
        self.assertEqual(result_message["source"], "result")

    def test_collect_traceloop_messages_crew_outputs_with_gen_ai_system(self):
        """
        Test _collect_traceloop_messages with CrewAI specific attributes when gen_ai.system is available
        """
        attributes = {
            "crewai.crew.tasks_output": "[TaskOutput(description='Task description', output='Task result')]",
            "crewai.crew.result": "Final crew execution result",
            "traceloop.entity.name": "oldvalue",
            "gen_ai.system": "crewai-agent",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 2)

        # Check the tasks output message
        tasks_message = messages[0]
        self.assertEqual(tasks_message["content"], "[TaskOutput(description='Task description', output='Task result')]")
        self.assertEqual(tasks_message["role"], "assistant")
        self.assertEqual(tasks_message["source"], "output")

        # Check the result message
        result_message = messages[1]
        self.assertEqual(result_message["content"], "Final crew execution result")
        self.assertEqual(result_message["role"], "assistant")
        self.assertEqual(result_message["source"], "result")

    def test_collect_traceloop_messages_entity_with_gen_ai_system(self):
        """
        Test that traceloop.entity.input and traceloop.entity.output messages are collected
        even when gen_ai.system is available
        """
        attributes = {
            "traceloop.entity.input": "input data",
            "traceloop.entity.output": "output data",
            "traceloop.entity.name": "my_entity",
            "gen_ai.system": "should-not-be-used",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899

        messages = self.llo_handler._collect_all_llo_messages(span, attributes)

        self.assertEqual(len(messages), 2)

        # Regular traceloop entity messages
        input_message = messages[0]
        self.assertEqual(input_message["content"], "input data")
        self.assertEqual(input_message["role"], "user")
        self.assertEqual(input_message["source"], "input")

        output_message = messages[1]
        self.assertEqual(output_message["content"], "output data")
        self.assertEqual(output_message["role"], "assistant")
        self.assertEqual(output_message["source"], "output")

    def test_emit_llo_attributes(self):
        """
        Test _emit_llo_attributes with consolidated event schema
        """
        attributes = {
            "gen_ai.prompt.0.content": "prompt content",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.completion.0.content": "completion content",
            "gen_ai.completion.0.role": "assistant",
            "traceloop.entity.input": "traceloop input",
            "traceloop.entity.name": "entity_name",
            "gen_ai.system": "openai",
            "gen_ai.agent.actual_output": "agent output",
            "crewai.crew.tasks_output": "tasks output",
            "crewai.crew.result": "crew result",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899
        span.instrumentation_scope = MagicMock()
        span.instrumentation_scope.name = "test.scope"

        self.llo_handler._emit_llo_attributes(span, attributes)

        # Verify that a single consolidated event was emitted
        self.event_logger_mock.emit.assert_called_once()
        emitted_event = self.event_logger_mock.emit.call_args[0][0]

        # Check event structure
        self.assertEqual(emitted_event.name, "test.scope")
        self.assertEqual(emitted_event.timestamp, span.end_time)
        self.assertEqual(emitted_event.trace_id, span.context.trace_id)
        self.assertEqual(emitted_event.span_id, span.context.span_id)
        self.assertEqual(emitted_event.trace_flags, span.context.trace_flags)

        # Check event body has input/output structure
        event_body = emitted_event.body
        self.assertIn("input", event_body)
        self.assertIn("output", event_body)
        self.assertIn("messages", event_body["input"])
        self.assertIn("messages", event_body["output"])

        # Check input messages
        input_messages = event_body["input"]["messages"]
        self.assertEqual(len(input_messages), 2)  # user prompt + traceloop input

        # Find user prompt message
        user_prompt = next((msg for msg in input_messages if msg["content"] == "prompt content"), None)
        self.assertIsNotNone(user_prompt)
        self.assertEqual(user_prompt["role"], "user")

        # Find traceloop input message
        traceloop_input = next((msg for msg in input_messages if msg["content"] == "traceloop input"), None)
        self.assertIsNotNone(traceloop_input)
        self.assertEqual(traceloop_input["role"], "user")

        # Check output messages
        output_messages = event_body["output"]["messages"]
        self.assertTrue(len(output_messages) >= 3)  # completion + agent output + crew outputs

        # Find completion message
        completion = next((msg for msg in output_messages if msg["content"] == "completion content"), None)
        self.assertIsNotNone(completion)
        self.assertEqual(completion["role"], "assistant")

        # Find agent output message
        agent_output = next((msg for msg in output_messages if msg["content"] == "agent output"), None)
        self.assertIsNotNone(agent_output)
        self.assertEqual(agent_output["role"], "assistant")

    def test_process_spans(self):
        """
        Test process_spans
        """
        attributes = {"gen_ai.prompt.0.content": "prompt content", "normal.attribute": "normal value"}

        span = self._create_mock_span(attributes)
        span.events = []  # No events initially

        with patch.object(self.llo_handler, "_emit_llo_attributes") as mock_emit, patch.object(
            self.llo_handler, "_filter_attributes"
        ) as mock_filter, patch.object(self.llo_handler, "process_span_events") as mock_process_events:

            filtered_attributes = {"normal.attribute": "normal value"}
            mock_filter.return_value = filtered_attributes

            result = self.llo_handler.process_spans([span])

            mock_emit.assert_called_once_with(span, attributes)
            mock_filter.assert_called_once_with(attributes)
            mock_process_events.assert_called_once_with(span)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], span)
            # Access the _attributes property that was set by the process_spans method
            self.assertEqual(result[0]._attributes, filtered_attributes)

    def test_process_spans_with_bounded_attributes(self):
        """
        Test process_spans with BoundedAttributes
        """
        from opentelemetry.attributes import BoundedAttributes

        bounded_attrs = BoundedAttributes(
            maxlen=10,
            attributes={"gen_ai.prompt.0.content": "prompt content", "normal.attribute": "normal value"},
            immutable=False,
            max_value_len=1000,
        )

        span = self._create_mock_span(bounded_attrs)

        with patch.object(self.llo_handler, "_emit_llo_attributes") as mock_emit, patch.object(
            self.llo_handler, "_filter_attributes"
        ) as mock_filter:

            filtered_attributes = {"normal.attribute": "normal value"}
            mock_filter.return_value = filtered_attributes

            result = self.llo_handler.process_spans([span])

            mock_emit.assert_called_once_with(span, bounded_attrs)
            mock_filter.assert_called_once_with(bounded_attrs)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], span)
            # Check that we got a BoundedAttributes instance
            self.assertIsInstance(result[0]._attributes, BoundedAttributes)
            # Check the underlying dictionary content
            self.assertEqual(dict(result[0]._attributes), filtered_attributes)

    def test_emit_llo_attributes_multiple_frameworks(self):
        """
        Test _emit_llo_attributes with attributes from multiple frameworks in a single span
        """
        attributes = {
            # Standard Gen AI
            "gen_ai.prompt.0.content": "Tell me about AI",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.completion.0.content": "AI is a field of computer science...",
            "gen_ai.completion.0.role": "assistant",
            # Traceloop
            "traceloop.entity.input": "What is machine learning?",
            "traceloop.entity.output": "Machine learning is a subset of AI...",
            # OpenLit
            "gen_ai.prompt": "Explain neural networks",
            "gen_ai.completion": "Neural networks are computing systems...",
            # OpenInference
            "input.value": "How do transformers work?",
            "output.value": "Transformers are a type of neural network architecture...",
            # CrewAI
            "crewai.crew.result": "Task completed successfully",
            "gen_ai.system": "multi-framework-test",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899
        span.instrumentation_scope = MagicMock()
        span.instrumentation_scope.name = "test.multi.framework"

        self.llo_handler._emit_llo_attributes(span, attributes)

        # Verify single consolidated event was emitted
        self.event_logger_mock.emit.assert_called_once()
        emitted_event = self.event_logger_mock.emit.call_args[0][0]

        # Check event metadata
        self.assertEqual(emitted_event.name, "test.multi.framework")
        self.assertEqual(emitted_event.timestamp, span.end_time)

        # Check event body structure
        event_body = emitted_event.body
        self.assertIn("input", event_body)
        self.assertIn("output", event_body)

        # Check that all input messages are present
        input_messages = event_body["input"]["messages"]
        input_contents = [msg["content"] for msg in input_messages]
        self.assertIn("Tell me about AI", input_contents)
        self.assertIn("What is machine learning?", input_contents)
        self.assertIn("Explain neural networks", input_contents)
        self.assertIn("How do transformers work?", input_contents)

        # Check that all output messages are present
        output_messages = event_body["output"]["messages"]
        output_contents = [msg["content"] for msg in output_messages]
        self.assertIn("AI is a field of computer science...", output_contents)
        self.assertIn("Machine learning is a subset of AI...", output_contents)
        self.assertIn("Neural networks are computing systems...", output_contents)
        self.assertIn("Transformers are a type of neural network architecture...", output_contents)
        self.assertIn("Task completed successfully", output_contents)

        # Verify role assignment
        for msg in input_messages:
            self.assertIn(msg["role"], ["user", "system"])
        for msg in output_messages:
            self.assertEqual(msg["role"], "assistant")

    def test_emit_llo_attributes_no_llo_attributes(self):
        """
        Test _emit_llo_attributes when there are no LLO attributes
        """
        attributes = {
            "normal.attribute": "value",
            "another.attribute": 123,
        }

        span = self._create_mock_span(attributes)
        span.instrumentation_scope = MagicMock()
        span.instrumentation_scope.name = "test.scope"

        self.llo_handler._emit_llo_attributes(span, attributes)

        # Should not emit any event when no LLO attributes are present
        self.event_logger_mock.emit.assert_not_called()

    def test_emit_llo_attributes_mixed_input_output(self):
        """
        Test _emit_llo_attributes with a mix of input and output messages
        """
        attributes = {
            "gen_ai.prompt.0.content": "system message",
            "gen_ai.prompt.0.role": "system",
            "gen_ai.prompt.1.content": "user message",
            "gen_ai.prompt.1.role": "user",
            "gen_ai.completion.0.content": "assistant response",
            "gen_ai.completion.0.role": "assistant",
            "input.value": "direct input",
            "output.value": "direct output",
            "gen_ai.system": "test-system",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899
        span.instrumentation_scope = MagicMock()
        span.instrumentation_scope.name = "test.scope"

        self.llo_handler._emit_llo_attributes(span, attributes)

        # Verify event was emitted
        self.event_logger_mock.emit.assert_called_once()
        emitted_event = self.event_logger_mock.emit.call_args[0][0]

        # Check event body structure
        event_body = emitted_event.body
        self.assertIn("input", event_body)
        self.assertIn("output", event_body)

        # Check input messages (system, user, direct input)
        input_messages = event_body["input"]["messages"]
        self.assertEqual(len(input_messages), 3)

        # Verify message roles
        input_roles = [msg["role"] for msg in input_messages]
        self.assertIn("system", input_roles)
        self.assertIn("user", input_roles)

        # Check output messages (assistant, direct output)
        output_messages = event_body["output"]["messages"]
        self.assertEqual(len(output_messages), 2)

        # Verify all output messages have assistant role
        for msg in output_messages:
            self.assertEqual(msg["role"], "assistant")

    def test_process_span_events(self):

        # Create span with events containing LLO attributes
        event_attributes = {
            "gen_ai.prompt": "event prompt",
            "normal.attribute": "keep this",
        }

        event = SpanEvent(
            name="test_event",
            attributes=event_attributes,
            timestamp=1234567890,
        )

        span = self._create_mock_span({})
        span.events = [event]
        span.instrumentation_scope = MagicMock()
        span.instrumentation_scope.name = "test.scope"

        with patch.object(self.llo_handler, "_emit_llo_attributes") as mock_emit:
            self.llo_handler.process_span_events(span)

            # Verify _emit_llo_attributes was called with event attributes
            mock_emit.assert_called_once_with(span, event_attributes, event_timestamp=1234567890)

            # Verify event attributes were filtered
            updated_event = span._events[0]
            self.assertIn("normal.attribute", updated_event.attributes)
            self.assertNotIn("gen_ai.prompt", updated_event.attributes)

    def test_process_span_events_no_events(self):
        """
        Test process_span_events when span has no events
        """
        span = self._create_mock_span({})
        span.events = None
        span._events = None  # Initialize _events attribute on the mock

        # Should handle gracefully without errors
        self.llo_handler.process_span_events(span)

        # _events should remain None (not set by process_span_events)
        # Since the span has no events, _events should not be changed
        self.assertIsNone(span._events)

    def test_emit_llo_attributes_with_event_timestamp(self):
        """
        Test _emit_llo_attributes uses event timestamp when provided
        """
        attributes = {
            "gen_ai.prompt": "test prompt",
            "gen_ai.system": "test",
        }

        span = self._create_mock_span(attributes)
        span.end_time = 1234567899
        span.instrumentation_scope = MagicMock()
        span.instrumentation_scope.name = "test.scope"

        event_timestamp = 9999999999

        self.llo_handler._emit_llo_attributes(span, attributes, event_timestamp=event_timestamp)

        # Verify event was emitted with event timestamp
        self.event_logger_mock.emit.assert_called_once()
        emitted_event = self.event_logger_mock.emit.call_args[0][0]
        self.assertEqual(emitted_event.timestamp, event_timestamp)  # Should use event timestamp, not span end time

    def test_collect_methods_message_format(self):
        """
        Test that all collect methods return messages in the expected format
        """
        # Test attributes covering all frameworks
        attributes = {
            "gen_ai.prompt.0.content": "prompt",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.completion.0.content": "response",
            "gen_ai.completion.0.role": "assistant",
            "traceloop.entity.input": "input",
            "gen_ai.prompt": "direct prompt",
            "input.value": "inference input",
        }

        span = self._create_mock_span(attributes)

        # Test each collect method returns proper message format
        prompt_messages = self.llo_handler._collect_all_llo_messages(span, attributes)
        for msg in prompt_messages:
            self.assertIn("content", msg)
            self.assertIn("role", msg)
            self.assertIn("source", msg)
            self.assertIsInstance(msg["content"], str)
            self.assertIsInstance(msg["role"], str)
            self.assertIsInstance(msg["source"], str)

        completion_messages = self.llo_handler._collect_all_llo_messages(span, attributes)
        for msg in completion_messages:
            self.assertIn("content", msg)
            self.assertIn("role", msg)
            self.assertIn("source", msg)

        traceloop_messages = self.llo_handler._collect_all_llo_messages(span, attributes)
        for msg in traceloop_messages:
            self.assertIn("content", msg)
            self.assertIn("role", msg)
            self.assertIn("source", msg)

        openlit_messages = self.llo_handler._collect_all_llo_messages(span, attributes)
        for msg in openlit_messages:
            self.assertIn("content", msg)
            self.assertIn("role", msg)
            self.assertIn("source", msg)

        openinference_messages = self.llo_handler._collect_all_llo_messages(span, attributes)
        for msg in openinference_messages:
            self.assertIn("content", msg)
            self.assertIn("role", msg)
            self.assertIn("source", msg)
