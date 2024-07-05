import ast
from BedrockInference import BedrockInferenceStream, get_tokens
import boto3
from botocore.config import Config
import json
import logging
import os
from SageMakerInference import SageMakerInferenceStream
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

                inputTokens = get_tokens(messages_text)
        else:
            inputTokens = get_tokens(body["inputs"])

        if bedrock_streaming.get_output_tokens() != 0:
            outputTokens = bedrock_streaming.get_output_tokens()
        else:
            outputTokens = get_tokens(response)

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
