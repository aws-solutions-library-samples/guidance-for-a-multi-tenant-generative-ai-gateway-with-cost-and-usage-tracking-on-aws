import ast
import base64
import boto3
from botocore.config import Config
import io
import json
from langchain_community.llms.bedrock import LLMInputOutputAdapter
from langchain_core.outputs import GenerationChunk
import logging
import math
import os
import time
import traceback
from typing import Dict

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

dynamodb = boto3.resource('dynamodb')
s3_client = boto3.client('s3')

bedrock_region = os.environ.get("BEDROCK_REGION", "us-east-1")
bedrock_url = os.environ.get("BEDROCK_URL", None)
iam_role = os.environ.get("IAM_ROLE", None)
streaming_table_name = os.environ.get("STREAMING_TABLE_NAME", None)
s3_bucket = os.environ.get("S3_BUCKET", None)
sagemaker_endpoints = os.environ.get("SAGEMAKER_ENDPOINTS", "") # If FMs are exposed through SageMaker
sagemaker_region = os.environ.get("SAGEMAKER_REGION", "us-east-1") # If FMs are exposed through SageMaker
sagemaker_url = os.environ.get("SAGEMAKER_URL", None) # If FMs are exposed through SageMaker

# Constants
GUARDRAILS_BODY_KEY = "amazon-bedrock-guardrailAssessment"

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

class SageMakerInferenceStream:
    def __init__(self, sagemaker_runtime, endpoint_name):
        self.sagemaker_runtime = sagemaker_runtime
        self.endpoint_name = endpoint_name
        # A buffered I/O stream to combine the payload parts:
        self.buff = io.BytesIO()
        self.read_pos = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def get_input_tokens(self):
        return self.input_tokens

    def get_output_tokens(self):
        return self.output_tokens

    def invoke_text_streaming(self, body, model_kwargs):
        try:
            request_body = {
                "inputs": body["inputs"],
                "parameters": model_kwargs
            }

            stream = self.stream(request_body)

            response = self.prepare_output_stream_messages_api(stream)

            self.input_tokens = _get_tokens(body["inputs"])

            return response

        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    def prepare_output_stream_messages_api(self, stream):
        tmp_response = ""
        for part in stream:
            tmp_response += part

        try:
            response = json.loads(tmp_response)
        except json.JSONDecodeError:
            # Invalid JSON, try to fix it
            if not tmp_response.endswith("}"):
                # Missing closing bracket
                tmp_response = tmp_response + "}"
            if not tmp_response.endswith("]"):
                # Uneven brackets
                tmp_response = tmp_response + "]"

            # Try again
            response = json.loads(tmp_response)

        response = response[0]["generated_text"]

        self.output_tokens = _get_tokens(response)

        return response

    def stream(self, request_body):
        # Gets a streaming inference response
        # from the specified model endpoint:
        response = self.sagemaker_runtime \
            .invoke_endpoint_with_response_stream(
            EndpointName=self.endpoint_name,
            Body=json.dumps(request_body),
            ContentType="application/json"
        )
        # Gets the EventStream object returned by the SDK:
        event_stream = response['Body']
        for event in event_stream:
            # Passes the contents of each payload part
            # to be concatenated:
            self._write(event['PayloadPart']['Bytes'])
            # Iterates over lines to parse whole JSON objects:
            for line in self._readlines():
                # Returns parts incrementally:
                yield line.decode("utf-8")

    # Writes to the buffer to concatenate the contents of the parts:
    def _write(self, content):
        self.buff.seek(0, io.SEEK_END)
        self.buff.write(content)

    # The JSON objects in buffer end with '\n'.
    # This method reads lines to yield a series of JSON objects:
    def _readlines(self):
        self.buff.seek(self.read_pos)
        for line in self.buff.readlines():
            self.read_pos += len(line)
            yield line[:-1]

def _get_bedrock_client():
    try:
        logger.info(f"Create new client\n  Using region: {bedrock_region}")
        session_kwargs = {"region_name": bedrock_region}
        client_kwargs = {**session_kwargs}

        retry_config = Config(
            region_name=bedrock_region,
            retries={
                "max_attempts": 10,
                "mode": "standard",
            },
        )
        session = boto3.Session(**session_kwargs)

        if iam_role is not None:
            logger.info(f"Using role: {iam_role}")
            sts = session.client("sts")

            response = sts.assume_role(
                RoleArn=str(iam_role),  #
                RoleSessionName="amazon-bedrock-assume-role"
            )

            client_kwargs = dict(
                aws_access_key_id=response['Credentials']['AccessKeyId'],
                aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                aws_session_token=response['Credentials']['SessionToken']
            )

        if bedrock_url:
            client_kwargs["endpoint_url"] = bedrock_url

        bedrock_client = session.client(
            service_name="bedrock-runtime",
            config=retry_config,
            **client_kwargs
        )

        logger.info("boto3 Bedrock client successfully created!")
        logger.info(bedrock_client._endpoint)
        return bedrock_client

    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error(stacktrace)

        raise e

def _get_sagemaker_client():
    try:
        logger.info(f"Create new client\n  Using region: {sagemaker_region}")
        session_kwargs = {"region_name": sagemaker_region}
        client_kwargs = {**session_kwargs}

        retry_config = Config(
            region_name=sagemaker_region,
            retries={
                "max_attempts": 10,
                "mode": "standard",
            },
        )
        session = boto3.Session(**session_kwargs)

        if iam_role is not None:
            logger.info(f"Using role: {iam_role}")
            sts = session.client("sts")

            response = sts.assume_role(
                RoleArn=str(iam_role),  #
                RoleSessionName="amazon-sagemaker-assume-role"
            )

            client_kwargs = dict(
                aws_access_key_id=response['Credentials']['AccessKeyId'],
                aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                aws_session_token=response['Credentials']['SessionToken']
            )

        if bedrock_url:
            client_kwargs["endpoint_url"] = sagemaker_url

        sagemaker_client = session.client(
            service_name="sagemaker-runtime",
            config=retry_config,
            **client_kwargs
        )

        logger.info("boto3 SageMaker client successfully created!")
        logger.info(sagemaker_client._endpoint)
        return sagemaker_client

    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error(stacktrace)

        raise e

def _get_tokens(string):
    logger.info("Counting approximation tokens")

    return math.floor(len(string) / 4)

def _read_json_event(event):
    try:
        request_json = event["request_json"]

        response = s3_client.get_object(Bucket=s3_bucket, Key=request_json)
        content = response['Body'].read()

        json_data = content.decode('utf-8')

        event = json.loads(json_data)

        s3_client.delete_object(Bucket=s3_bucket, Key=request_json)

        return event
    except Exception as e:
        stacktrace = traceback.format_exc()

        logger.error(stacktrace)

        raise e

def _read_sagemaker_endpoints():
    if not sagemaker_endpoints:
        return {}

    try:
        endpoints = json.loads(sagemaker_endpoints)
    except json.JSONDecodeError:
        try:
            endpoints = ast.literal_eval(sagemaker_endpoints)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Error: Invalid format for SAGEMAKER_ENDPOINTS: {e}")
    else:
        if not isinstance(endpoints, dict):
            raise ValueError("Error: SAGEMAKER_ENDPOINTS is not a dictionary")

    return endpoints

def bedrock_handler(event: Dict) -> Dict:
    try:
        bedrock_client = _get_bedrock_client()

        model_id = event["queryStringParameters"]['model_id']
        model_arn = event["queryStringParameters"].get('model_arn', None)
        request_id = event['queryStringParameters']['request_id']
        messages_api = event["headers"].get("messages_api", "false")
        api_key = event["headers"]["x-api-key"]

        logger.info(f"Model ID: {model_id}")
        logger.info(f"Request ID: {request_id}")

        body = json.loads(event["body"])
        model_kwargs = body.get("parameters", {})
        additional_model_fields = body.get("additional_model_fields", {})
        logger.info(f"Input body: {body}")

        bedrock_streaming = BedrockInferenceStream(
            bedrock_client=bedrock_client,
            model_id=model_id,
            model_arn=model_arn,
            messages_api=messages_api
        )

        response = "".join(chunk.text for chunk in bedrock_streaming.invoke_text_streaming(body, model_kwargs, additional_model_fields))
        logger.info(f"Answer: {response}")

        if messages_api.lower() in ["true"]:
            if bedrock_streaming.get_input_tokens() != 0:
                inputTokens = bedrock_streaming.get_input_tokens()
            else:
                messages_text = ""

                if "system" in model_kwargs:
                    messages_text += f"{model_kwargs['system']}\n"

                for message in body["inputs"]:
                    messages_text += f"{message['content']}\n"

                inputTokens = _get_tokens(messages_text)
        else:
            inputTokens = _get_tokens(body["inputs"])

        if bedrock_streaming.get_output_tokens() != 0:
            outputTokens = bedrock_streaming.get_output_tokens()
        else:
            outputTokens = _get_tokens(response)

        item = {
            "composite_pk": f"{request_id}_{api_key}",
            "request_id": request_id,
            "api_key": api_key,
            "status": 200,
            "generated_text": response,
            "inputTokens": inputTokens,
            "outputTokens": outputTokens,
            "model_id": model_id,
            "ttl": int(time.time()) + 2 * 60
        }

        logger.info(f"Streaming answer: {item}")

        connections = dynamodb.Table(streaming_table_name)
        connections.put_item(Item=item)

        logger.info(f"Put item: {response}")

        return {"statusCode": 200, "body": response}

    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error(stacktrace)

        model_id = event.get("queryStringParameters", {}).get('model_id', None)
        request_id = event.get("queryStringParameters", {}).get('request_id', None)

        api_key = event["headers"]["x-api-key"]

        if request_id is not None:
            item = {
                "composite_pk": f"{request_id}_{api_key}",
                "request_id": request_id,
                "api_key": api_key,
                "status": 500,
                "generated_text": stacktrace,
                "model_id": model_id,
                "ttl": int(time.time()) + 2 * 60
            }

            connections = dynamodb.Table(streaming_table_name)
            connections.put_item(Item=item)

            logger.info(f"Put exception item: {stacktrace}")

        return {"statusCode": 500, "body": json.dumps([{"generated_text": stacktrace}])}

def sagemaker_handler(event: Dict) -> Dict:
    try:
        sagemaker_client = _get_sagemaker_client()

        model_id = event["queryStringParameters"]['model_id']
        request_id = event['queryStringParameters']['request_id']

        api_key = event["headers"]["x-api-key"]

        logger.info(f"Model ID: {model_id}")
        logger.info(f"Request ID: {request_id}")

        body = json.loads(event["body"])
        model_kwargs = body.get("parameters", {})

        logger.info(f"Input body: {body}")

        endpoints = _read_sagemaker_endpoints()
        endpoint_name = endpoints[model_id]

        sagemaker_streaming = SageMakerInferenceStream(sagemaker_client, endpoint_name)

        response = sagemaker_streaming.invoke_text_streaming(body, model_kwargs)
        logger.info(f"Answer: {response}")

        item = {
            "composite_pk": f"{request_id}_{api_key}",
            "request_id": request_id,
            "api_key": api_key,
            "status": 200,
            "generated_text": response,
            "inputs": body["inputs"],
            "inputTokens": sagemaker_streaming.get_input_tokens(),
            "outputTokens": sagemaker_streaming.get_output_tokens(),
            "model_id": model_id,
            "ttl": int(time.time()) + 2 * 60
        }

        connections = dynamodb.Table(streaming_table_name)
        connections.put_item(Item=item)

        logger.info(f"Put item: {response}")

        return {"statusCode": 200, "body": response}

    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error(stacktrace)

        model_id = event.get("queryStringParameters", {}).get('model_id', None)
        request_id = event.get("queryStringParameters", {}).get('request_id', None)

        api_key = event["headers"]["x-api-key"]

        if request_id is not None:
            item = {
                "composite_pk": f"{request_id}_{api_key}",
                "request_id": request_id,
                "api_key": api_key,
                "status": 500,
                "generated_text": stacktrace,
                "model_id": model_id,
                "ttl": int(time.time()) + 2 * 60
            }

            connections = dynamodb.Table(streaming_table_name)
            connections.put_item(Item=item)

            logger.info(f"Put exception item: {stacktrace}")

        return {"statusCode": 500, "body": json.dumps([{"generated_text": stacktrace}])}

def lambda_handler(event: Dict, context) -> Dict:
    event = _read_json_event(event)

    logger.info(event)

    model_id = event["queryStringParameters"]['model_id']

    endpoints = _read_sagemaker_endpoints()

    if model_id in endpoints:
        return sagemaker_handler(event)
    else:
        return bedrock_handler(event)
