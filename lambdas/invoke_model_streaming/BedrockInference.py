import base64
import json
from langchain_community.llms.bedrock import LLMInputOutputAdapter
from langchain_core.outputs import GenerationChunk
import logging
import math
import traceback

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

GUARDRAILS_BODY_KEY = "amazon-bedrock-guardrailAssessment"

def get_tokens(string):
    logger.info("Counting approximation tokens")

    return math.floor(len(string) / 4)

class BedrockInferenceStream:
    def __init__(self, bedrock_client, model_id, model_arn=None, messages_api="false"):
        self.bedrock_client = bedrock_client
        self.model_id = model_id
        self.model_arn = model_arn
        self.messages_api = messages_api
        self.input_tokens = 0
        self.output_tokens = 0

    """
    Decode base64-encoded documents in the input messages.

    Args:
        messages (list): A list of message dictionaries.

    Returns:
        list: The updated list of message dictionaries with decoded images.
    """
    def _decode_documents(self, messages):
        for item in messages:
            if 'content' in item:
                for content_item in item['content']:
                    if 'document' in content_item and 'bytes' in content_item['document']['source']:
                        encoded_document = content_item['document']['source']['bytes']
                        base64_bytes = encoded_document.encode('utf-8')
                        document_bytes = base64.b64decode(base64_bytes)
                        content_item['document']['source']['bytes'] = document_bytes
        return messages

    """
    Decode base64-encoded images in the input messages.

    Args:
        messages (list): A list of message dictionaries.

    Returns:
        list: The updated list of message dictionaries with decoded images.
    """
    def _decode_images(self, messages):
        for item in messages:
            if 'content' in item:
                for content_item in item['content']:
                    if 'image' in content_item and 'bytes' in content_item['image']['source']:
                        encoded_image = content_item['image']['source']['bytes']
                        base64_bytes = encoded_image.encode('utf-8')
                        image_bytes = base64.b64decode(base64_bytes)
                        content_item['image']['source']['bytes'] = image_bytes
        return messages

    def get_input_tokens(self):
        return self.input_tokens

    def get_output_tokens(self):
        return self.output_tokens

    def invoke_text_streaming(self, body, model_kwargs:dict = dict(), additional_model_fields:dict = dict()):
        try:
            provider = self.model_id.split(".")[0]

            if self.messages_api.lower() in ["true"]:
                system = [{"text": model_kwargs["system"]}] if "system" in model_kwargs else list()

                if "system" in model_kwargs:
                    del model_kwargs["system"]

                messages = self._decode_documents(body["inputs"])
                messages = self._decode_images(messages)

                modelId = self.model_arn if self.model_arn is not None else self.model_id

                response = self.bedrock_client.converse_stream(
                    modelId=modelId,
                    messages=messages,
                    system=system,
                    inferenceConfig=model_kwargs,
                    additionalModelRequestFields=additional_model_fields
                )

                return self.prepare_output_stream(provider, response, messages_api=True)
            else:
                request_body = LLMInputOutputAdapter.prepare_input(
                    provider=provider,
                    prompt=body["inputs"],
                    model_kwargs=model_kwargs
                )

                request_body = json.dumps(request_body)

                return self.stream(request_body)

        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    def prepare_output_stream(self, provider, response, stop=None, messages_api=False):
        if messages_api:
            stream = response.get("stream")
        else:
            stream = response.get("body")

        if not stream:
            return

        if messages_api:
            output_key = "message"
        else:
            output_key = LLMInputOutputAdapter.provider_to_output_key_map.get(provider, "")

        if not output_key:
            raise ValueError(
                f"Unknown streaming response output key for provider: {provider}"
            )

        for event in stream:
            if messages_api:
                if 'contentBlockDelta' in event:
                    chunk_obj = event['contentBlockDelta']
                    if "delta" in chunk_obj and "text" in chunk_obj["delta"]:
                        chk = GenerationChunk(
                            text=chunk_obj["delta"]["text"],
                            generation_info=dict(
                                finish_reason=chunk_obj.get("stop_reason", None),
                            ),
                        )
                        yield chk

                if "metadata" in event and "usage" in event["metadata"]:
                    usage = event["metadata"]["usage"]
                    if "inputTokens" in usage:
                        self.input_tokens += usage["inputTokens"]
                    if "outputTokens" in usage:
                        self.output_tokens += usage["outputTokens"]

            else:
                chunk = event.get("chunk")
                if not chunk:
                    continue

                chunk_obj = json.loads(chunk.get("bytes").decode())

                if provider == "cohere" and (
                        chunk_obj["is_finished"] or chunk_obj[output_key] == "<EOS_TOKEN>"
                ):
                    return

                elif (
                        provider == "mistral"
                        and chunk_obj.get(output_key, [{}])[0].get("stop_reason", "") == "stop"
                ):
                    return

                elif messages_api and (chunk_obj.get("type") == "content_block_stop"):
                    return

                if messages_api and chunk_obj.get("type") in (
                        "message_start",
                        "content_block_start",
                        "content_block_delta",
                ):
                    if chunk_obj.get("type") == "content_block_delta":
                        if not chunk_obj["delta"]:
                            chk = GenerationChunk(text="")
                        else:
                            chk = GenerationChunk(
                                text=chunk_obj["delta"]["text"],
                                generation_info=dict(
                                    finish_reason=chunk_obj.get("stop_reason", None),
                                ),
                            )
                        yield chk
                    else:
                        continue
                else:
                    if messages_api:
                        if chunk_obj["type"] == "message_start" and "message" in chunk_obj and "usage" in chunk_obj["message"]:
                            if "input_tokens" in chunk_obj["message"]["usage"]:
                                self.input_tokens += int(chunk_obj["message"]["usage"]["input_tokens"])
                            if "output_tokens" in chunk_obj["message"]["usage"]:
                                self.output_tokens += int(chunk_obj["message"]["usage"]["output_tokens"])
                        if chunk_obj["type"] == "message_delta" and "usage" in chunk_obj:
                            if "input_tokens" in chunk_obj["usage"]:
                                self.input_tokens += int(chunk_obj["usage"]["input_tokens"])
                            if "output_tokens" in chunk_obj["usage"]:
                                self.output_tokens += int(chunk_obj["usage"]["output_tokens"])

                    # chunk obj format varies with provider
                    yield GenerationChunk(
                        text=(
                            chunk_obj[output_key]
                            if provider != "mistral"
                            else chunk_obj[output_key][0]["text"]
                        ),
                        generation_info={
                            GUARDRAILS_BODY_KEY: (
                                chunk_obj.get(GUARDRAILS_BODY_KEY)
                                if GUARDRAILS_BODY_KEY in chunk_obj
                                else None
                            ),
                        },
                    )

    def stream(self, request_body):
        try:
            provider = self.model_id.split(".")[0]

            modelId = self.model_arn if self.model_arn is not None else self.model_id

            response = self.bedrock_client.invoke_model_with_response_stream(
                body=request_body,
                modelId=modelId,
                accept="application/json",
                contentType="application/json",
            )
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

        if self.messages_api.lower() in ["true"]:
            for chunk in self.prepare_output_stream(provider, response, messages_api=True):
                yield chunk
        else:
            for chunk in self.prepare_output_stream(provider, response, messages_api=False):
                yield chunk