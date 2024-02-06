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
fields 
message.team_id as team_id,
message.requestId as request_id,
message.region as region,
message.model_id as model_id,
message.inputTokens as input_tokens,
message.outputTokens as output_tokens,
message.height as height,
message.width as width,
message.steps as steps
| filter level = "INFO"
"""

def process_event(event):
    try:
        if "date" in event:
            date = event["date"]
        else:
            date = datetime.datetime.now(pytz.UTC) - datetime.timedelta(days=1)
            date = date.strftime("%Y-%m-%d")

        # querying the cloudwatch logs from the API
        query_results_api = run_query(QUERY_API, log_group_name_api, date)
        df_bedrock_cost_tracking = results_to_df(query_results_api, date)

        print(df_bedrock_cost_tracking.head())

        # Apply the calculate_cost function to the DataFrame
        df_bedrock_cost_tracking[["date", "input_tokens", "output_tokens", "input_cost", "output_cost", "invocations"]] = df_bedrock_cost_tracking.apply(
            calculate_cost, axis=1, result_type="expand"
        )

        # aggregate cost for each model_id
        df_bedrock_cost_tracking_aggregated = df_bedrock_cost_tracking.groupby(["team_id", "model_id"]).sum()[
            ["date", "input_tokens", "output_tokens", "input_cost", "output_cost", "invocations"]
        ]

        logger.info(df_bedrock_cost_tracking_aggregated.to_string())

        csv_buffer = StringIO()
        df_bedrock_cost_tracking_aggregated.to_csv(csv_buffer)

        file_name = f"{date}.csv"

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
