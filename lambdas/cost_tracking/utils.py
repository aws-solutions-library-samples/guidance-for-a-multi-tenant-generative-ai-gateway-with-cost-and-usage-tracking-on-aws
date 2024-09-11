import boto3
import datetime
import json
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

def _is_in_model_list(model_id, model_list):
    if model_id in model_list:
        return True
    else:
        parts = model_id.split('.')
        for i in range(len(parts), 0, -1):
            partial_id = '.'.join(parts[-i:])
            if partial_id in model_list:
                return True

    return False

def _read_model_list(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            config = json.load(f)

        return config
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error(stacktrace)

        raise e

def get_model_pricing(model_id, MODEL_PRICES):
    matched = [v for k, v in MODEL_PRICES.items() if model_id in k]
    if matched:
        return matched[0]
    else:
        parts = model_id.split('.')
        for i in range(len(parts), 0, -1):
            partial_id = '.'.join(parts[-i:])
            matched = [v for k, v in MODEL_PRICES.items() if partial_id in k]
            if matched:
                return matched[0]

        return None

def run_query(query, log_group_name, date=None):
    cloudwatch = boto3.client("logs")

    max_retries = 5

    if date is None:
        date = datetime.datetime.now(pytz.UTC) - datetime.timedelta(days=1)
    else:
        date = datetime.datetime.strptime(date, "%Y-%m-%d")

    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = date.replace(hour=23, minute=59, second=59, microsecond=0)

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

def model_price_embeddings(model_list, row):
    input_token_count = float(row["input_tokens"]) if "input_tokens" in row else 0.0
    output_token_count = float(row["output_tokens"]) if "output_tokens" in row else 0.0

    model_id = row["model_id"]

    # get model pricing for each region
    model_pricing = get_model_pricing(model_id, model_list)

    # calculate costs of prompt and completion
    input_cost = input_token_count * model_pricing["input_cost"] / 1000
    output_cost = output_token_count * model_pricing["output_cost"] / 1000

    return input_token_count, output_token_count, input_cost, output_cost

def model_price_image(model_list, row):
    height = float(row["height"]) if "height" in row else 0.0
    width = float(row["width"]) if "width" in row else 0.0
    steps = float(row["steps"]) if "steps" in row else 0.0

    model_id = row["model_id"]

    # get model pricing from utils
    model_pricing = get_model_pricing(model_id, model_list)

    if width <= 512 and height <= 512:
        size = "512x512"
    else:
        size = "larger"

    model_pricing = get_model_pricing(size, model_pricing)

    if steps > 50:
        price = model_pricing["premium"]
    else:
        price = model_pricing["standard"]

    return 0.0, 0.0, 0.0, price

def model_price_text(model_list, row):
    input_token_count = float(row["input_tokens"]) if "input_tokens" in row else 0.0
    output_token_count = float(row["output_tokens"]) if "output_tokens" in row else 0.0

    model_id = row["model_id"]

    # get model pricing for each region
    model_pricing = get_model_pricing(model_id, model_list)

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

def calculate_cost(row, error_buffer):
    try:
        model_id = row["model_id"]

        models = _read_model_list("./models.json")

        region = row["region"] if "region" in row else "us-east-1"

        model_list = models[region]

        if _is_in_model_list(model_id, list(model_list["text"].keys())):
            input_token_count, output_token_count, input_cost, output_cost = model_price_text(model_list["text"], row)
        elif _is_in_model_list(model_id, list(model_list["embeddings"].keys())):
            input_token_count, output_token_count, input_cost, output_cost = model_price_embeddings(model_list["embeddings"], row)
        elif _is_in_model_list(model_id, list(model_list["image"].keys())):
            input_token_count, output_token_count, input_cost, output_cost = model_price_image(model_list["image"], row)
        else:
            raise Exception(f"Unknown model: {model_id}")

        return input_token_count, output_token_count, input_cost, output_cost, 1
    except Exception as e:
        stacktrace = traceback.format_exc()

        logger.error(f"Error processing row: {row}\n{stacktrace}")

        error_buffer.write(f"Row:\n{row}\n\n Stacktrace:\n{stacktrace}\n\n")

        return None, None, None, None, None
