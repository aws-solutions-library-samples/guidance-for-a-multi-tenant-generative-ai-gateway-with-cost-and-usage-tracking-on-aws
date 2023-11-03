from aws_lambda_powertools import Logger
import boto3
from botocore.config import Config
import json
from langchain.llms.bedrock import LLMInputOutputAdapter
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

bedrock_region = os.environ.get("BEDROCK_REGION", "us-east-1")
bedrock_role = os.environ.get("BEDROCK_ROLE", None)
bedrock_url = os.environ.get("BEDROCK_URL", None)

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

        if bedrock_role is not None:
            logger.info(f"Using role: {bedrock_role}")
            sts = session.client("sts")

            response = sts.assume_role(
                RoleArn=str(bedrock_role),  #
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
            service_name="bedrock",
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

def _get_tokens(string):
    logger.info("Counting approximation tokens")

    return math.floor(len(string)/4)


def lambda_handler(event, context):
    try:
        if "team_id" in event["headers"] and event["headers"]["team_id"] is not None and event["headers"]["team_id"] != "":
            bedrock_client = _get_bedrock_client()

            logger.info(event)
            model_id = event["queryStringParameters"]['model_id']
            request_id = event['requestContext']['requestId']
            team_id = event["headers"]["team_id"]

            if "embeddings" in event["headers"] and event["headers"]["embeddings"] in ["True", "true"]:
                embeddings = True
            else:
                embeddings = False

            provider = model_id.split(".")[0]

            logger.info(f"Model ID: {model_id}")
            logger.info(f"Provider: {provider}")
            logger.info(f"Request ID: {request_id}")

            body = json.loads(event["body"])

            logger.info(f"Input body: {body}")

            model_kwargs = body["parameters"] if "parameters" in body else {}

            if embeddings:
                request_body = {**model_kwargs, "inputText": body["inputs"]}
                request_body = json.dumps(request_body)

                response = bedrock_client.invoke_model(
                    body=request_body,
                    modelId=model_id,
                    accept="application/json",
                    contentType="application/json",
                )

                response_body = json.loads(response.get("body").read())
                response = response_body.get("embedding")

                results = {"statusCode": 200, "body": json.dumps([{"embedding": response}])}
            else:
                request_body = LLMInputOutputAdapter.prepare_input(provider, body["inputs"], model_kwargs)

                request_body = json.dumps(request_body)

                response = bedrock_client.invoke_model(
                    body=request_body,
                    modelId=model_id,
                    accept="application/json",
                    contentType="application/json"
                )

                response = LLMInputOutputAdapter.prepare_output(provider, response)

                results = {"statusCode": 200, "body": json.dumps([{"generated_text": response}])}

            cloudwatch_logger.info({
                "team_id": team_id,
                "requestId": request_id,
                "model_id": model_id,
                "inputTokens": _get_tokens(body["inputs"]),
                "outputTokens": _get_tokens(response),
            })

            return results
        else:
            logger.error("Bad Request: Header 'team_id' is missing")

            return {"statusCode": 400, "body": "Bad Request"}

    except Exception as e:
        stacktrace = traceback.format_exc()

        logger.error(stacktrace)
        return {"statusCode": 500, "body": json.dumps([{"generated_text": stacktrace}])}