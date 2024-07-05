import ast
from aws_lambda_powertools import Logger
from BedrockInference import BedrockInference, get_tokens
import boto3
from botocore.config import Config
import datetime
import json
import logging
import os
from SageMakerInference import SageMakerInference
import time
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
logs_table_name = os.environ.get("LOGS_TABLE_NAME", None)
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

"""
Return the json list of enabled SageMaker Endpoints

Returns:
    dict: json list of enabled SageMaker Endpoints
"""
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

"""
Save model logs in CloudWatch and DynamoDB

Args:
    logs (dict): logs generated by the model
    
Raises:
        Exception: If an error occurs during the cloudwatch or dynamodb save.
"""
def _store_logs(logs):
    try:
        cloudwatch_logger.info(logs)

        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        logs["date"] = formatted_datetime
        logs["ttl"] = int(time.time()) + 86400

        logs_table_connection = dynamodb.Table(logs_table_name)
        logs_table_connection.put_item(Item=logs)
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error(stacktrace)

        raise e

"""
This function handles inference requests for Bedrock models.

Args:
    event (dict): The event object containing the request details.

Returns:
    dict: A dictionary containing the response status code and body.
"""
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
                "inputTokens": get_tokens(body["inputs"]),
                "outputTokens": get_tokens(response),
                "height": None,
                "width": None,
                "steps": None
            }

            _store_logs(logs)

        elif embeddings_image:
            logger.info("Request type: embeddings-image")
            response = bedrock_inference.invoke_embeddings_image(body, model_kwargs)
            results = {"statusCode": 200, "body": json.dumps([{"embedding": response}])}
            logs = {
                "team_id": team_id,
                "requestId": request_id,
                "region": bedrock_region,
                "model_id": model_id + "-image",
                "inputTokens": get_tokens(body["inputs"]),
                "outputTokens": get_tokens(response),
                "height": None,
                "width": None,
                "steps": None
            }

            _store_logs(logs)
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

            _store_logs(logs)
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

                _store_logs(logs)
        return results

    else:
        logger.info("Check streaming request")
        connections = dynamodb.Table(streaming_table_name)
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

            _store_logs(logs)
        else:
            results = {"statusCode": 200, "body": json.dumps([{"request_id": custom_request_id}])}

        return results

"""
This function handles inference requests for SageMaker models.

Args:
    event (dict): The event object containing the request details.

Returns:
    dict: A dictionary containing the response status code and body.
"""
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

            _store_logs(logs)
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

                _store_logs(logs)

        return results

    else:
        logger.info("Check streaming request")
        connections = dynamodb.Table(streaming_table_name)
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
