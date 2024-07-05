import json
import logging
import math
import traceback

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

"""
Return an approximation of tokens in a string
Args:
    string (str): Input string

Returns:
    int: Number of approximated tokens
"""
def get_tokens(string):
    logger.info("Counting approximation tokens")

    return math.floor(len(string) / 4)

"""
This class handles inference requests for SageMaker models.
"""
class SageMakerInference:
    """
    Initialize the SageMakerInference instance.

    Args:
        sagemaker_client (boto3.client): The SageMaker client instance.
        endpoint_name (str): The name of the SageMaker endpoint.
        messages_api (str, optional): Whether to use the messages API. Defaults to "false".
    """
    def __init__(self, sagemaker_client, endpoint_name, messages_api="false"):
        self.sagemaker_client = sagemaker_client
        self.endpoint_name = endpoint_name
        self.messages_api = messages_api
        self.input_tokens = 0
        self.output_tokens = 0

    """
    Get the number of input tokens.

    Returns:
        int: The number of input tokens.
    """
    def get_input_tokens(self):
        return self.input_tokens

    """
    Get the number of output tokens.

    Returns:
        int: The number of output tokens.
    """
    def get_output_tokens(self):
        return self.output_tokens

    """
    Invoke the SageMaker model to generate embeddings for text inputs.

    Args:
        body (dict): The request body containing the input text.
        model_kwargs (dict): Additional model parameters.

    Returns:
        list: A list of embeddings for the input text.

    Raises:
        Exception: If an error occurs during the inference process.
    """
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

            self.input_tokens = get_tokens(body["inputs"][list(body["inputs"].keys())[0]])

            return response["embedding"]
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    """
    Invoke the SageMaker model to generate text from prompts.

    Args:
        body (dict): The request body containing the input prompts.
        model_kwargs (dict): Additional model parameters.

    Returns:
        str: The generated text.

    Raises:
        Exception: If an error occurs during the inference process.
    """
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

                self.input_tokens = get_tokens(body["inputs"])
                self.output_tokens = get_tokens(response)

                return response
            else:
                self.input_tokens = get_tokens(body["inputs"])
                self.output_tokens = get_tokens(response[0]["generated_text"])

                return response[0]["generated_text"]
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e