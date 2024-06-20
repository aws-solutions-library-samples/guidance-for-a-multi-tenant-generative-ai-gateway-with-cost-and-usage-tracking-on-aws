import ast
from aws_lambda_powertools import Logger
import base64
import boto3
from botocore.config import Config
import json
from langchain_community.llms.bedrock import LLMInputOutputAdapter
import logging
import math
import os
import traceback

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

cloudwatch_logger = Logger()

lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')
s3_client = boto3.client('s3')

bedrock_region = os.environ.get("BEDROCK_REGION", "us-east-1")
bedrock_url = os.environ.get("BEDROCK_URL", None)
iam_role = os.environ.get("IAM_ROLE", None)
lambda_streaming = os.environ.get("LAMBDA_STREAMING", None)
table_name = os.environ.get("TABLE_NAME", None)
s3_bucket = os.environ.get("S3_BUCKET", None)
sagemaker_endpoints = os.environ.get("SAGEMAKER_ENDPOINTS", "") # If FMs are exposed through SageMaker
sagemaker_region = os.environ.get("SAGEMAKER_REGION", "us-east-1") # If FMs are exposed through SageMaker
sagemaker_url = os.environ.get("SAGEMAKER_URL", None) # If FMs are exposed through SageMaker

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

class BedrockInference:
    def __init__(self, bedrock_client, model_id, model_arn=None, messages_api="false"):
        self.bedrock_client = bedrock_client
        self.model_id = model_id
        self.model_arn = model_arn
        self.messages_api = messages_api
        self.input_tokens = 0
        self.output_tokens = 0

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

    def invoke_embeddings(self, body, model_kwargs):
        try:
            provider = self.model_id.split(".")[0]

            if provider == "cohere":
                if "input_type" not in model_kwargs.keys():
                    model_kwargs["input_type"] = "search_document"
                if isinstance(body["inputs"], str):
                    body["inputs"] = [body["inputs"]]

                request_body = {**model_kwargs, "texts": body["inputs"]}
            else:
                request_body = {**model_kwargs, "inputText": body["inputs"]}

            request_body = json.dumps(request_body)

            modelId = self.model_arn if self.model_arn is not None else self.model_id

            response = self.bedrock_client.invoke_model(
                body=request_body,
                modelId=modelId,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            if provider == "cohere":
                response = response_body.get("embeddings")[0]
            else:
                response = response_body.get("embedding")

            return response
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    def invoke_embeddings_image(self, body, model_kwargs):
        try:
            provider = self.model_id.split(".")[0]

            request_body = {**model_kwargs, "inputImage": body["inputs"]}

            request_body = json.dumps(request_body)

            modelId = self.model_arn if self.model_arn is not None else self.model_id

            response = self.bedrock_client.invoke_model(
                body=request_body,
                modelId=modelId,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            if provider == "cohere":
                response = response_body.get("embeddings")[0]
            else:
                response = response_body.get("embedding")

            return response
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    def invoke_image(self, body, model_kwargs):
        try:
            provider = self.model_id.split(".")[0]

            if provider == "stability":
                request_body = {**model_kwargs, "text_prompts": body["text_prompts"]}

                height = model_kwargs["height"] if "height" in model_kwargs else 512
                width = model_kwargs["width"] if "width" in model_kwargs else 512
                steps = model_kwargs["steps"] if "steps" in model_kwargs else 50
            else:
                request_body = {**model_kwargs, "textToImageParams": body["textToImageParams"]}

                height = model_kwargs["imageGenerationConfig"]["height"] if "height" in model_kwargs[
                    "imageGenerationConfig"] else 512
                width = model_kwargs["imageGenerationConfig"]["width"] if "width" in model_kwargs[
                    "imageGenerationConfig"] else 512

                if "quality" in model_kwargs["imageGenerationConfig"]:
                    if model_kwargs["imageGenerationConfig"]["quality"] == "standard":
                        steps = 50
                    else:
                        steps = 51
                else:
                    steps = 50

            request_body = json.dumps(request_body)

            modelId = self.model_arn if self.model_arn is not None else self.model_id

            response = self.bedrock_client.invoke_model(
                body=request_body,
                modelId=modelId,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            if provider == "stability":
                response = {"artifacts": response_body.get("artifacts")}
            else:
                response = {"images": response_body.get("images")}

            return response, height, width, steps
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    def invoke_text(self, body, model_kwargs: dict = dict(), additional_model_fields: dict = dict()):
        try:
            provider = self.model_id.split(".")[0]
            is_messages_api = self.messages_api.lower() in ["true"]

            if is_messages_api:
                system = [{"text": model_kwargs["system"]}] if "system" in model_kwargs else list()

                if "system" in model_kwargs:
                    del model_kwargs["system"]

                messages = self._decode_images(body["inputs"])

                response = self.bedrock_client.converse(
                    modelId=self.model_id,
                    messages=messages,
                    system=system,
                    inferenceConfig=model_kwargs,
                    additionalModelRequestFields=additional_model_fields
                )

                output_message = response['output']['message']

                self.input_tokens = response['usage']['inputTokens']
                self.output_tokens = response['usage']['outputTokens']

                tmp_response = ""

                for content in output_message['content']:
                    tmp_response += content['text'] + " "

                response = tmp_response.rstrip()
            else:
                request_body = LLMInputOutputAdapter.prepare_input(
                    provider=provider,
                    prompt=body["inputs"],
                    model_kwargs=model_kwargs
                )

                request_body = json.dumps(request_body)
                model_id = self.model_arn or self.model_id

                response = self.bedrock_client.invoke_model(
                    body=request_body,
                    modelId=model_id,
                    accept="application/json",
                    contentType="application/json"
                )

                response = LLMInputOutputAdapter.prepare_output(provider, response)
                response = response["text"]

                self.input_tokens = _get_tokens(body["inputs"])
                self.output_tokens = _get_tokens(response)

            return response
        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error(stacktrace)

            raise e

class SageMakerInference:
    def __init__(self, sagemaker_client, endpoint_name, messages_api="false"):
        self.sagemaker_client = sagemaker_client
        self.endpoint_name = endpoint_name
        self.messages_api = messages_api
        self.input_tokens = 0
        self.output_tokens = 0

    def get_input_tokens(self):
        return self.input_tokens

    def get_output_tokens(self):
        return self.output_tokens

    ## TBD implementation
    def invoke_embeddings(self, body, model_kwargs):
        try:
            if "InferenceComponentName" in model_kwargs:
                inference_component = model_kwargs.pop("InferenceComponentName")
            else:
                inference_component = None

            if isinstance(body["inputs"], dict):
                # If body["inputs"] is a dictionary, merge it with model_kwargs
                request_data = {**body["inputs"], **model_kwargs}
            else:
                # If body["inputs"] is not a dictionary, use the original format
                request_data = {
                    "inputs": body["inputs"],
                    "parameters": model_kwargs
                }

            request_body = json.dumps(request_data)

            if inference_component:
                response = self.sagemaker_client.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="application/json",
                    Body=request_body,
                    InferenceComponentName=inference_component
                )
            else:
                response = self.sagemaker_client.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="application/json",
                    Body=request_body
                )

            response = json.loads(response['Body'].read().decode())

            self.input_tokens = _get_tokens(body["inputs"][list(body["inputs"].keys())[0]])

            return response["embedding"]
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    def invoke_text(self, body, model_kwargs):
        try:
            if "InferenceComponentName" in model_kwargs:
                inference_component = model_kwargs.pop("InferenceComponentName")
            else:
                inference_component = None

            is_messages_api = self.messages_api.lower() in ["true"]

            if is_messages_api:
                request_data = {"messages": body["inputs"], **model_kwargs}
            else:
                if isinstance(body["inputs"], dict):
                    # If body["inputs"] is a dictionary, merge it with model_kwargs
                    request_data = {**body["inputs"], **model_kwargs}
                else:
                    # If body["inputs"] is not a dictionary, use the original format
                    request_data = {
                        "inputs": body["inputs"],
                        "parameters": model_kwargs
                    }

            request_body = json.dumps(request_data)

            if inference_component:
                response = self.sagemaker_client.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="application/json",
                    Body=request_body,
                    InferenceComponentName=inference_component
                )
            else:
                response = self.sagemaker_client.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="application/json",
                    Body=request_body
                )

            response = json.loads(response['Body'].read().decode())

            if is_messages_api:
                response = response["choices"][0]["message"]["content"].strip()

                self.input_tokens = _get_tokens(body["inputs"])
                self.output_tokens = _get_tokens(response)

                return response
            else:
                self.input_tokens = _get_tokens(body["inputs"])
                self.output_tokens = _get_tokens(response[0]["generated_text"])

                return response[0]["generated_text"]
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

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

def bedrock_handler(event):
    logger.info("Bedrock Endpoint")

    model_id = event["queryStringParameters"]["model_id"]
    model_arn = event["queryStringParameters"].get("model_arn")
    team_id = event["headers"]["team_id"]
    api_key = event["headers"]["x-api-key"]

    bedrock_client = _get_bedrock_client()
    custom_request_id = event["queryStringParameters"].get("requestId")
    messages_api = event["headers"].get("messages_api", "false")

    bedrock_inference = BedrockInference(
        bedrock_client=bedrock_client,
        model_id=model_id,
        model_arn=model_arn,
        messages_api=messages_api
    )

    if custom_request_id is None:
        request_id = event["requestContext"]["requestId"]
        streaming = event["headers"].get("streaming", "false")
        embeddings = event["headers"].get("type", "").lower() == "embeddings"
        embeddings_image = event["headers"].get("type", "").lower() == "embeddings-image"
        image = event["headers"].get("type", "").lower() == "image"

        logger.info(f"Model ID: {model_id}")
        logger.info(f"Request ID: {request_id}")

        body = json.loads(event["body"])
        model_kwargs = body.get("parameters", {})
        additional_model_fields = body.get("additional_model_fields", {})

        if embeddings:
            logger.info("Request type: embeddings")
            response = bedrock_inference.invoke_embeddings(body, model_kwargs)
            results = {"statusCode": 200, "body": json.dumps([{"embedding": response}])}
            logs = {
                "team_id": team_id,
                "requestId": request_id,
                "region": bedrock_region,
                "model_id": model_id,
                "inputTokens": _get_tokens(body["inputs"]),
                "outputTokens": _get_tokens(response),
                "height": None,
                "width": None,
                "steps": None
            }
            cloudwatch_logger.info(logs)

        elif embeddings_image:
            logger.info("Request type: embeddings-image")
            response = bedrock_inference.invoke_embeddings_image(body, model_kwargs)
            results = {"statusCode": 200, "body": json.dumps([{"embedding": response}])}
            logs = {
                "team_id": team_id,
                "requestId": request_id,
                "region": bedrock_region,
                "model_id": model_id + "-image",
                "inputTokens": _get_tokens(body["inputs"]),
                "outputTokens": _get_tokens(response),
                "height": None,
                "width": None,
                "steps": None
            }
            cloudwatch_logger.info(logs)

        elif image:
            logger.info("Request type: image")
            response, height, width, steps = bedrock_inference.invoke_image(body, model_kwargs)
            results = {"statusCode": 200, "body": json.dumps([response])}
            logs = {
                "team_id": team_id,
                "requestId": request_id,
                "region": bedrock_region,
                "model_id": model_id,
                "inputTokens": None,
                "outputTokens": None,
                "height": height,
                "width": width,
                "steps": steps
            }
            cloudwatch_logger.info(logs)

        else:
            logger.info("Request type: text")

            if streaming.lower() in ["true"] and custom_request_id is None:
                logger.info("Send streaming request")
                event["queryStringParameters"]["request_id"] = request_id
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=f"{request_id}.json",
                    Body=json.dumps(event).encode("utf-8")
                )
                lambda_client.invoke(
                    FunctionName=lambda_streaming,
                    InvocationType="Event",
                    Payload=json.dumps({"request_json": f"{request_id}.json"})
                )
                results = {"statusCode": 200, "body": json.dumps([{"request_id": request_id}])}

            else:
                response = bedrock_inference.invoke_text(body, model_kwargs, additional_model_fields)
                results = {"statusCode": 200, "body": json.dumps([{"generated_text": response}])}
                logs = {
                    "team_id": team_id,
                    "requestId": request_id,
                    "region": bedrock_region,
                    "model_id": model_id,
                    "inputTokens": bedrock_inference.get_input_tokens(),
                    "outputTokens": bedrock_inference.get_output_tokens(),
                    "height": None,
                    "width": None,
                    "steps": None
                }
                cloudwatch_logger.info(logs)

        return results

    else:
        logger.info("Check streaming request")
        connections = dynamodb.Table(table_name)
        response = connections.get_item(Key={"composite_pk": f"{custom_request_id}_{api_key}"})

        if "Item" in response:
            response = response["Item"]
            results = {
                "statusCode": response["status"],
                "body": json.dumps([{"generated_text": response["generated_text"]}])
            }
            connections.delete_item(Key={"composite_pk": f"{custom_request_id}_{api_key}"})
            logs = {
                "team_id": team_id,
                "requestId": custom_request_id,
                "region": bedrock_region,
                "model_id": response.get("model_id"),
                "inputTokens": int(response.get("inputTokens", 0)),
                "outputTokens": int(response.get("outputTokens", 0)),
                "height": None,
                "width": None,
                "steps": None
            }
            cloudwatch_logger.info(logs)
        else:
            results = {"statusCode": 200, "body": json.dumps([{"request_id": custom_request_id}])}

        return results

def sagemaker_handler(event):
    logger.info("SageMaker Endpoint")

    model_id = event["queryStringParameters"]["model_id"]
    team_id = event["headers"]["team_id"]
    api_key = event["headers"]["x-api-key"]

    sagemaker_client = _get_sagemaker_client()

    messages_api = event["headers"].get("messages_api", "false")
    custom_request_id = event["queryStringParameters"].get("requestId")

    endpoints = _read_sagemaker_endpoints()
    endpoint_name = endpoints[model_id]
    sagemaker_inference = SageMakerInference(sagemaker_client, endpoint_name, messages_api)

    if custom_request_id is None:
        request_id = event["requestContext"]["requestId"]
        streaming = event["headers"].get("streaming", "false")
        embeddings = event["headers"].get("type", "").lower() == "embeddings"

        logger.info(f"Model ID: {model_id}")
        logger.info(f"Request ID: {request_id}")

        body = json.loads(event["body"])
        model_kwargs = body.get("parameters", {})

        if embeddings:
            response = sagemaker_inference.invoke_embeddings(body, model_kwargs)
            results = {"statusCode": 200, "body": json.dumps([{"embedding": response}])}

            logs = {
                "team_id": team_id,
                "requestId": request_id,
                "region": sagemaker_region,
                "model_id": model_id,
                "inputTokens": sagemaker_inference.get_input_tokens(),
                "outputTokens": 0,
                "height": None,
                "width": None,
                "steps": None
            }
            cloudwatch_logger.info(logs)
        else:
            logger.info("Request type: text")

            if streaming.lower() in ["true"] and custom_request_id is None:
                logger.info("Send streaming request")
                event["queryStringParameters"]["request_id"] = request_id
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=f"{request_id}.json",
                    Body=json.dumps(event).encode("utf-8")
                )
                lambda_client.invoke(
                    FunctionName=lambda_streaming,
                    InvocationType="Event",
                    Payload=json.dumps({"request_json": f"{request_id}.json"})
                )
                results = {"statusCode": 200, "body": json.dumps([{"request_id": request_id}])}
            else:
                response = sagemaker_inference.invoke_text(body, model_kwargs)
                results = {"statusCode": 200, "body": json.dumps([{"generated_text": response}])}
                logs = {
                    "team_id": team_id,
                    "requestId": request_id,
                    "region": sagemaker_region,
                    "model_id": model_id,
                    "inputTokens": sagemaker_inference.get_input_tokens(),
                    "outputTokens": sagemaker_inference.get_output_tokens(),
                    "height": None,
                    "width": None,
                    "steps": None
                }
                cloudwatch_logger.info(logs)

        return results

    else:
        logger.info("Check streaming request")
        connections = dynamodb.Table(table_name)
        response = connections.get_item(Key={"composite_pk": f"{custom_request_id}_{api_key}"})

        if "Item" in response:
            response = response["Item"]
            results = {
                "statusCode": response["status"],
                "body": json.dumps([{"generated_text": response["generated_text"]}])
            }
            connections.delete_item(Key={"composite_pk": f"{custom_request_id}_{api_key}"})
            logs = {
                "team_id": team_id,
                "requestId": custom_request_id,
                "region": sagemaker_region,
                "model_id": response.get("model_id"),
                "inputTokens": int(response.get("inputTokens", 0)),
                "outputTokens": int(response.get("outputTokens", 0)),
                "height": None,
                "width": None,
                "steps": None
            }
            cloudwatch_logger.info(logs)
        else:
            results = {"statusCode": 200, "body": json.dumps([{"request_id": custom_request_id}])}

        return results

def lambda_handler(event, context):
    try:
        logger.info(event)

        team_id = event["headers"].get("team_id")
        if not team_id:
            logger.error("Bad Request: Header 'team_id' is missing")
            return {"statusCode": 400, "body": "Bad Request"}

        model_id = event["queryStringParameters"]["model_id"]
        endpoints = _read_sagemaker_endpoints()

        if model_id in endpoints:
            return sagemaker_handler(event)
        else:
            return bedrock_handler(event)

    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error(stacktrace)
        return {"statusCode": 500, "body": json.dumps([{"generated_text": stacktrace}])}
