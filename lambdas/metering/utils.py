import boto3
import datetime
import logging
import pandas as pd
import pytz
import time
import traceback

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

MODEL_PRICES_TEXT = {
    "amazon.titan-text-express-v1": {"input_cost": 0.0013, "output_cost": 0.0017},
    "amazon.titan-embed-text-v1": {"input_cost": 0.0001, "output_cost": 0},
    "ai21.j2-ultra-v1": {"input_cost": 0.0188, "output_cost": 0.0125},
    "ai21.j2-mid-v1": {"input_cost": 0.0125, "output_cost": 0.0188},
    "anthropic.claude-instant-v1": {"input_cost": 0.00163, "output_cost": 0.00551},
    "anthropic.claude-v1": {"input_cost": 0.01102, "output_cost": 0.03268},
    "anthropic.claude-v2": {"input_cost": 0.01102, "output_cost": 0.03268},
    "cohere.command-text-v14": {"input_cost": 0.0015, "output_cost": 0.0020},
}

MODEL_PRICES_IMAGE = {
    "stability.stable-diffusion-xl": {
        "512x512": {
            "standard": 0.018,
            "premium": 0.036
        },
        "larger": {
           "standard": 0.036,
           "premium": 0.072
        }
    }
}

def get_model_pricing(model_id, MODEL_PRICES):
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

def model_price_image(row):
    height = float(row["height"]) if "height" in row else 0.0
    width = float(row["width"]) if "width" in row else 0.0
    steps = float(row["steps"]) if "steps" in row else 0.0

    model_id = row["model_id"]

    # get model pricing from utils
    model_pricing = get_model_pricing(model_id, MODEL_PRICES_IMAGE)

    if width <= 512 and height <= 512:
        size = "512x512"
    else:
        size = "larger"

    model_pricing = get_model_pricing(size, model_pricing)

    if steps > 51:
        price = model_pricing["premium"]
    else:
        price = model_pricing["standard"]

    return 0.0, 0.0, 0.0, price

def model_price_text(row):
    input_token_count = float(row["input_tokens"]) if "input_tokens" in row else 0.0
    output_token_count = float(row["output_tokens"]) if "output_tokens" in row else 0.0

    model_id = row["model_id"]

    # get model pricing from utils
    model_pricing = get_model_pricing(model_id, MODEL_PRICES_TEXT)

    # calculate costs of prompt and completion
    input_cost = input_token_count * model_pricing["input_cost"] / 1000
    output_cost = output_token_count * model_pricing["output_cost"] / 1000

    return input_token_count, output_token_count, input_cost, output_cost

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
        model_id = row["model_id"]

        if model_id in list(MODEL_PRICES_TEXT.keys()):
            input_token_count, output_token_count, input_cost, output_cost = model_price_text(row)
        elif model_id in list(MODEL_PRICES_IMAGE.keys()):
            input_token_count, output_token_count, input_cost, output_cost = model_price_image(row)
        else:
            input_token_count, output_token_count, input_cost, output_cost = 0.0, 0.0, 0.0, 0.0

        return input_token_count, output_token_count, input_cost, output_cost, 1
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error(stacktrace)

        raise e
