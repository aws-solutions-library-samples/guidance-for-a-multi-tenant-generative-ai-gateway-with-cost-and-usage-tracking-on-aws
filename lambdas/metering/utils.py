import time
import boto3
import datetime
import pandas as pd
import pytz

MODEL_PRICES = {
    "amazon.titan-text-express-v1": {"input_cost": 0.0013, "output_cost": 0.0017},
    "amazon.titan-embed-text-v1": {"input_cost": 0.0001, "output_cost": 0},
    "stability.stable-diffusion-xl": {"input_cost": 0.018, "output_cost": 0.036},
    "ai21.j2-ultra-v1": {"input_cost": 0.0188, "output_cost": 0.0125},
    "ai21.j2-mid-v1": {"input_cost": 0.0125, "output_cost": 0.0188},
    "anthropic.claude-instant-v1": {"input_cost": 0.00163, "output_cost": 0.00551},
    "anthropic.claude-v1": {"input_cost": 0.01102, "output_cost": 0.03268},
    "anthropic.claude-v2": {"input_cost": 0.01102, "output_cost": 0.03268},
    "cohere.command-text-v14": {"input_cost": 0.0015, "output_cost": 0.0020},
}

def get_model_pricing(model_id):
    matched = [v for k, v in MODEL_PRICES.items() if model_id in k]
    if not matched:
        return None
    else:
        return matched[0]

def run_query(query, log_group_name):
    cloudwatch = boto3.client("logs")

    max_retries = 5

    yesterday = datetime.datetime.now(pytz.UTC) - datetime.timedelta(days=1)

    start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    end = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)

    response = cloudwatch.start_query(
        logGroupName=log_group_name,
        startTime=int(start.timestamp() * 1000),
        endTime=int(end.timestamp() * 1000),
        queryString=query,
    )

    query_id = response["queryId"]

    retry_count = 0

    while True:
        response = cloudwatch.get_query_results(queryId=query_id)

        if response["results"] or retry_count == max_retries:
            break

        time.sleep(2)
        retry_count += 1

    return response["results"]

def results_to_df(results):
    column_names = set()
    rows = []

    for result in results:
        row = {
            item["field"]: item["value"]
            for item in result
            if "@ptr" not in item["field"]
        }
        column_names.update(row.keys())
        rows.append(row)

    df = pd.DataFrame(rows, columns=list(column_names))
    return df

def calculate_cost(row):
    try:
        input_token_count = float(row["input_tokens"])
        output_token_count = float(row["output_tokens"])
        model_id = row["model_id"]

        # get model pricing from utils
        model_pricing = get_model_pricing(model_id)

        # calculate costs of prompt and completion
        input_cost = input_token_count * model_pricing["input_cost"] / 1000
        output_cost = output_token_count * model_pricing["output_cost"] / 1000

        return input_token_count, output_token_count, input_cost, output_cost, 1
    except (ValueError, KeyError):
        # Handle cases where data is not in the expected format
        return None, None, None, None
