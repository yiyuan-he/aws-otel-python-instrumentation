# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import inspect
import io
import json
from typing import Any, Dict, Optional

from amazon.opentelemetry.distro._aws_attribute_keys import (
    AWS_BEDROCK_AGENT_ID,
    AWS_BEDROCK_DATA_SOURCE_ID,
    AWS_BEDROCK_GUARDRAIL_ARN,
    AWS_BEDROCK_GUARDRAIL_ID,
    AWS_BEDROCK_KNOWLEDGE_BASE_ID,
)
from amazon.opentelemetry.distro._aws_span_processing_util import GEN_AI_REQUEST_MAX_TOKENS, GEN_AI_REQUEST_MODEL, GEN_AI_REQUEST_TEMPERATURE, GEN_AI_REQUEST_TOP_P, GEN_AI_RESPONSE_FINISH_REASONS, GEN_AI_SYSTEM, GEN_AI_USAGE_INPUT_TOKENS, GEN_AI_USAGE_OUTPUT_TOKENS
from opentelemetry.instrumentation.botocore.extensions.types import (
    _AttributeMapT,
    _AwsSdkCallContext,
    _AwsSdkExtension,
    _BotoResultT,
)
from opentelemetry.trace.span import Span
from botocore.response import StreamingBody

_AGENT_ID: str = "agentId"
_KNOWLEDGE_BASE_ID: str = "knowledgeBaseId"
_DATA_SOURCE_ID: str = "dataSourceId"
_GUARDRAIL_ID: str = "guardrailId"
_GUARDRAIL_ARN: str = "guardrailArn"
_MODEL_ID: str = "modelId"
_AWS_BEDROCK_SYSTEM: str = "aws_bedrock"


class _BedrockAgentOperation(abc.ABC):
    """
    We use subclasses and operation names to handle specific Bedrock Agent operations.
    - Only operations involving Agent, DataSource, or KnowledgeBase resources are supported.
    - Operations without these specified resources are not covered.
    - When an operation involves multiple resources (e.g., AssociateAgentKnowledgeBase),
      we map it to one resource based on some judgement classification of rules.

    For detailed API documentation on Bedrock Agent operations, visit:
    https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Agents_for_Amazon_Bedrock.html
    """

    request_attributes: Optional[Dict[str, str]] = None
    response_attributes: Optional[Dict[str, str]] = None

    @classmethod
    @abc.abstractmethod
    def operation_names(cls):
        pass


class _AgentOperation(_BedrockAgentOperation):
    """
    This class covers BedrockAgent API related to <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_Agent.html">Agents</a>,
    and extracts agent-related attributes.
    """

    request_attributes = {
        AWS_BEDROCK_AGENT_ID: _AGENT_ID,
    }
    response_attributes = {
        AWS_BEDROCK_AGENT_ID: _AGENT_ID,
    }

    @classmethod
    def operation_names(cls):
        return [
            "CreateAgentActionGroup",
            "CreateAgentAlias",
            "DeleteAgentActionGroup",
            "DeleteAgentAlias",
            "DeleteAgent",
            "DeleteAgentVersion",
            "GetAgentActionGroup",
            "GetAgentAlias",
            "GetAgent",
            "GetAgentVersion",
            "ListAgentActionGroups",
            "ListAgentAliases",
            "ListAgentKnowledgeBases",
            "ListAgentVersions",
            "PrepareAgent",
            "UpdateAgentActionGroup",
            "UpdateAgentAlias",
            "UpdateAgent",
        ]


class _KnowledgeBaseOperation(_BedrockAgentOperation):
    """
    This class covers BedrockAgent API related to <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_KnowledgeBase.html">KnowledgeBases</a>,
    and extracts knowledge base-related attributes.

    Note: The 'CreateDataSource' operation does not have a 'dataSourceId' in the context,
    but it always comes with a 'knowledgeBaseId'. Therefore, we categorize it under 'knowledgeBaseId' operations.
    """

    request_attributes = {
        AWS_BEDROCK_KNOWLEDGE_BASE_ID: _KNOWLEDGE_BASE_ID,
    }
    response_attributes = {
        AWS_BEDROCK_KNOWLEDGE_BASE_ID: _KNOWLEDGE_BASE_ID,
    }

    @classmethod
    def operation_names(cls):
        return [
            "AssociateAgentKnowledgeBase",
            "CreateDataSource",
            "DeleteKnowledgeBase",
            "DisassociateAgentKnowledgeBase",
            "GetAgentKnowledgeBase",
            "GetKnowledgeBase",
            "ListDataSources",
            "UpdateAgentKnowledgeBase",
        ]


class _DataSourceOperation(_BedrockAgentOperation):
    """
    This class covers BedrockAgent API related to <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_DataSource.html">DataSources</a>,
    and extracts data source-related attributes.
    """

    request_attributes = {
        AWS_BEDROCK_KNOWLEDGE_BASE_ID: _KNOWLEDGE_BASE_ID,
        AWS_BEDROCK_DATA_SOURCE_ID: _DATA_SOURCE_ID,
    }
    response_attributes = {
        AWS_BEDROCK_DATA_SOURCE_ID: _DATA_SOURCE_ID,
    }

    @classmethod
    def operation_names(cls):
        return ["DeleteDataSource", "GetDataSource", "UpdateDataSource"]


# _OPERATION_NAME_TO_CLASS_MAPPING maps operation names to their corresponding classes
# by iterating over all subclasses of _BedrockAgentOperation and extract operations
# by calling operation_names() function.
_OPERATION_NAME_TO_CLASS_MAPPING = {
    op_name: op_class
    for op_class in [_KnowledgeBaseOperation, _DataSourceOperation, _AgentOperation]
    for op_name in op_class.operation_names()
    if inspect.isclass(op_class) and issubclass(op_class, _BedrockAgentOperation) and not inspect.isabstract(op_class)
}


class _BedrockAgentExtension(_AwsSdkExtension):
    """
    This class is an extension for <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Agents_for_Amazon_Bedrock.html">
    Agents for Amazon Bedrock</a>.

    This class primarily identify three types of resource based operations: _AgentOperation, _KnowledgeBaseOperation,
    and _DataSourceOperation. We only support operations that are related to the resource
    and where the context contains the resource ID.
    """

    def __init__(self, call_context: _AwsSdkCallContext):
        super().__init__(call_context)
        self._operation_class = _OPERATION_NAME_TO_CLASS_MAPPING.get(call_context.operation)

    def extract_attributes(self, attributes: _AttributeMapT):
        if self._operation_class is None:
            return
        for attribute_key, request_param_key in self._operation_class.request_attributes.items():
            request_param_value = self._call_context.params.get(request_param_key)
            if request_param_value:
                attributes[attribute_key] = request_param_value

    def on_success(self, span: Span, result: _BotoResultT):
        if self._operation_class is None:
            return

        for attribute_key, response_param_key in self._operation_class.response_attributes.items():
            response_param_value = result.get(response_param_key)
            if response_param_value:
                span.set_attribute(
                    attribute_key,
                    response_param_value,
                )


class _BedrockAgentRuntimeExtension(_AwsSdkExtension):
    """
    This class is an extension for <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Agents_for_Amazon_Bedrock_Runtime.html">
    Agents for Amazon Bedrock Runtime</a>.
    """

    def extract_attributes(self, attributes: _AttributeMapT):
        agent_id = self._call_context.params.get(_AGENT_ID)
        if agent_id:
            attributes[AWS_BEDROCK_AGENT_ID] = agent_id

        knowledge_base_id = self._call_context.params.get(_KNOWLEDGE_BASE_ID)
        if knowledge_base_id:
            attributes[AWS_BEDROCK_KNOWLEDGE_BASE_ID] = knowledge_base_id


class _BedrockExtension(_AwsSdkExtension):
    """
    This class is an extension for <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Amazon_Bedrock.html">Bedrock</a>.
    """

    # pylint: disable=no-self-use
    def on_success(self, span: Span, result: _BotoResultT):
        # _GUARDRAIL_ID can only be retrieved from the response, not from the request
        guardrail_id = result.get(_GUARDRAIL_ID)
        if guardrail_id:
            span.set_attribute(
                AWS_BEDROCK_GUARDRAIL_ID,
                guardrail_id,
            )

        guardrail_arn = result.get(_GUARDRAIL_ARN)
        if guardrail_arn:
            span.set_attribute(
                AWS_BEDROCK_GUARDRAIL_ARN,
                guardrail_arn,
            )


class _BedrockRuntimeExtension(_AwsSdkExtension):
    """
    This class is an extension for <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Amazon_Bedrock_Runtime.html">
    Amazon Bedrock Runtime</a>.
    """

    def extract_attributes(self, attributes: _AttributeMapT):
        attributes[GEN_AI_SYSTEM] = _AWS_BEDROCK_SYSTEM

        model_id = self._call_context.params.get(_MODEL_ID)
        if model_id:
            attributes[GEN_AI_REQUEST_MODEL] = model_id

            # Get the request body if it exists
            body = self._call_context.params.get('body')
            if body:
                try:
                    request_body = json.loads(body)
                    if "amazon.titan" in model_id:
                        self._extract_titan_attributes(attributes, request_body)
                except json.JSONDecodeError:
                    print("Error: Unable to parse the body as JSON")

    def _extract_titan_attributes(self, attributes, request_body):
        config = request_body.get('textGenerationConfig', {})
        self._set_if_not_none(attributes, GEN_AI_REQUEST_MAX_TOKENS, config.get('maxTokenCount'))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TEMPERATURE, config.get('temperature'))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TOP_P, config.get("topP"))

    @staticmethod
    def _set_if_not_none(attributes, key, value):
        if value is not None:
            attributes[key] = value

    def on_success(self, span: Span, result: Dict[str, Any]):
        model_id = self._call_context.params.get(_MODEL_ID)

        if not model_id:
            return

        if 'body' in result and isinstance(result['body'], StreamingBody):
            original_body = None
            try:
                original_body = result['body']
                body_content = original_body.read()
                
                # Use one stream for telemetry
                stream = io.BytesIO(body_content)
                telemetry_content = stream.read()
                response_body = json.loads(telemetry_content.decode('utf-8'))
                if "amazon.titan" in model_id:
                    self._handle_amazon_titan_response(span, response_body)
                
                # Replenish stream for downstream application use
                new_stream = io.BytesIO(body_content)
                result['body'] = StreamingBody(new_stream, len(body_content))
                    
            except json.JSONDecodeError:
                print("Error: Unable to parse the response body as JSON")
            except Exception as e:
                print(f"Error processing response: {str(e)}")
            finally:
                if original_body is not None:
                    original_body.close()

    def _handle_amazon_titan_response(self, span: Span, response_body: Dict[str, Any]):
        if "inputTextTokenCount" in response_body:
            span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, response_body['inputTextTokenCount'])

            result = response_body['results'][0]
            # TODO: Need null check for result here?
            if 'tokenCount' in result:
                span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, result['tokenCount'])
            if 'completionReason' in result:
                span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [result['completionReason']])

