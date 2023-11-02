import boto3
from botocore.config import Config
import json
import logging
import os
import traceback

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

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

def lambda_handler(event, context):
    try:
        bedrock_client = _get_bedrock_client()

        logger.info(event)

        response = bedrock_client.list_foundation_models()

        return {"statusCode": 200, "body": json.dumps([response])}

    except Exception as e:
        stacktrace = traceback.format_exc()

        logger.error(stacktrace)
        return {"statusCode": 500, "body": json.dumps([{"generated_text": stacktrace}])}