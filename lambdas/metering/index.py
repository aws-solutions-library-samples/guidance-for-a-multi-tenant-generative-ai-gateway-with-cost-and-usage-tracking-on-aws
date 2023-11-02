import boto3
import datetime
from io import StringIO
import logging
import os
import pytz
import traceback
from utils import run_query, results_to_df, calculate_cost

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

log_group_name_api = os.environ.get("LOG_GROUP_API", None)
s3_bucket = os.environ.get("S3_BUCKET", None)

s3_resource = boto3.resource('s3')

QUERY_API = """
fields message.team_id as team_id, message.request_id as request_id, message.model_id as model_id, message.inputTokens as input_tokens, message.outputTokens as output_tokens
| filter level = "INFO"
"""

def process_event(event):
    try:
        # querying the cloudwatch logs from the API
        query_results_api = run_query(QUERY_API, log_group_name_api)
        df_bedrock_metering = results_to_df(query_results_api)

        # Apply the calculate_cost function to the DataFrame
        df_bedrock_metering[["input_tokens", "output_tokens", "input_cost", "output_cost", "invocations"]] = df_bedrock_metering.apply(
            calculate_cost, axis=1, result_type="expand"
        )

        # aggregate cost for each model_id
        df_bedrock_metering_aggregated = df_bedrock_metering.groupby(["team_id", "model_id"]).sum()[
            ["input_tokens", "output_tokens", "invocations", "input_cost", "output_cost"]
        ]

        logger.info(df_bedrock_metering_aggregated.to_string())

        csv_buffer = StringIO()
        df_bedrock_metering_aggregated.to_csv(csv_buffer)

        yesterday = datetime.datetime.now(pytz.UTC) - datetime.timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        file_name = f"{yesterday_str}.csv"

        s3_resource.Object(s3_bucket, file_name).put(Body=csv_buffer.getvalue())
    except Exception as e:
        stacktrace = traceback.format_exc()

        logger.error(stacktrace)
        raise e

def lambda_handler(event, context):
    try:
        process_event(event)
        return {"statusCode": 200, "body": "OK"}
    except Exception as e:
        stacktrace = traceback.format_exc()

        logger.error(stacktrace)
        return {"statusCode": 500, "body": str(e)}
