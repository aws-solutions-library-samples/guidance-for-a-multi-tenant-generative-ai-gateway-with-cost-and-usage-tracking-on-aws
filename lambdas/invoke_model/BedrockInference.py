import base64
import json
from langchain_community.llms.bedrock import LLMInputOutputAdapter
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
This class handles inference requests for Bedrock models.
"""
class BedrockInference:
    """
    Initialize the BedrockInference instance.

    Args:
        bedrock_client (boto3.client): The Bedrock client instance.
        model_id (str): The ID of the model to use.
        model_arn (str, optional): The ARN of the model to use. Defaults to None.
        messages_api (str, optional): Whether to use the messages API. Defaults to "false".
    """
    def __init__(self, bedrock_client, model_id, model_arn=None, messages_api="false"):
        self.bedrock_client = bedrock_client
        self.model_id = model_id
        self.model_arn = model_arn
        self.messages_api = messages_api
        self.input_tokens = 0
        self.output_tokens = 0

    """
    Decode base64-encoded documents in the input messages.

    Args:
        messages (list): A list of message dictionaries.

    Returns:
        list: The updated list of message dictionaries with decoded images.
    """
    def _decode_documents(self, messages):
        for item in messages:
            if 'content' in item:
                for content_item in item['content']:
                    if 'document' in content_item and 'bytes' in content_item['document']['source']:
                        encoded_document = content_item['document']['source']['bytes']
                        base64_bytes = encoded_document.encode('utf-8')
                        document_bytes = base64.b64decode(base64_bytes)
                        content_item['document']['source']['bytes'] = document_bytes
        return messages

    """
    Decode base64-encoded images in the input messages.

    Args:
        messages (list): A list of message dictionaries.

    Returns:
        list: The updated list of message dictionaries with decoded images.
    """
    def _decode_images(self, messages):
        for item in messages:
            if 'content' in item:
                for content_item in item['content']:
                    if 'image' in content_item and 'bytes' in content_item['image']['source']:
                        encoded_image = content_item['image']['source']['bytes']
                        base64_bytes = encoded_image.encode('utf-8')
                        image_bytes = base64.b64decode(base64_bytes)
                        content_item['image']['source']['bytes'] = image_bytes
        return messages

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
    Invoke the Bedrock model to generate embeddings for text inputs.

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
            provider = self.model_id.split(".")[0]

            if provider == "cohere":
                if "input_type" not in model_kwargs.keys():
                    model_kwargs["input_type"] = "search_document"
                if isinstance(body["inputs"], str):
                    body["inputs"] = [body["inputs"]]

                request_body = {**model_kwargs, "texts": body["inputs"]}
            else:
                request_body = {**model_kwargs, "inputText": body["inputs"]}

            request_body = json.dumps(request_body)

            modelId = self.model_arn if self.model_arn is not None else self.model_id

            response = self.bedrock_client.invoke_model(
                body=request_body,
                modelId=modelId,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            if provider == "cohere":
                response = response_body.get("embeddings")[0]
            else:
                response = response_body.get("embedding")

            return response
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    """
    Invoke the Bedrock model to generate embeddings for image inputs.

    Args:
        body (dict): The request body containing the input image.
        model_kwargs (dict): Additional model parameters.

    Returns:
        list: A list of embeddings for the input image.

    Raises:
        Exception: If an error occurs during the inference process.
    """
    def invoke_embeddings_image(self, body, model_kwargs):
        try:
            provider = self.model_id.split(".")[0]

            request_body = {**model_kwargs, "inputImage": body["inputs"]}

            request_body = json.dumps(request_body)

            modelId = self.model_arn if self.model_arn is not None else self.model_id

            response = self.bedrock_client.invoke_model(
                body=request_body,
                modelId=modelId,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            if provider == "cohere":
                response = response_body.get("embeddings")[0]
            else:
                response = response_body.get("embedding")

            return response
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    """
    Invoke the Bedrock model to generate images from text prompts.

    Args:
        body (dict): The request body containing the text prompts.
        model_kwargs (dict): Additional model parameters.

    Returns:
        dict: A dictionary containing the generated images and their dimensions.
        int: The height of the generated images.
        int: The width of the generated images.
        int: The number of steps used to generate the images.

    Raises:
        Exception: If an error occurs during the inference process.
    """
    def invoke_image(self, body, model_kwargs):
        try:
            provider = self.model_id.split(".")[0]

            if provider == "stability":
                request_body = {**model_kwargs, "text_prompts": body["text_prompts"]}

                height = model_kwargs["height"] if "height" in model_kwargs else 512
                width = model_kwargs["width"] if "width" in model_kwargs else 512
                steps = model_kwargs["steps"] if "steps" in model_kwargs else 50
            else:
                request_body = {**model_kwargs, "textToImageParams": body["textToImageParams"]}

                height = model_kwargs["imageGenerationConfig"]["height"] if "height" in model_kwargs[
                    "imageGenerationConfig"] else 512
                width = model_kwargs["imageGenerationConfig"]["width"] if "width" in model_kwargs[
                    "imageGenerationConfig"] else 512

                if "quality" in model_kwargs["imageGenerationConfig"]:
                    if model_kwargs["imageGenerationConfig"]["quality"] == "standard":
                        steps = 50
                    else:
                        steps = 51
                else:
                    steps = 50

            request_body = json.dumps(request_body)

            modelId = self.model_arn if self.model_arn is not None else self.model_id

            response = self.bedrock_client.invoke_model(
                body=request_body,
                modelId=modelId,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            if provider == "stability":
                response = {"artifacts": response_body.get("artifacts")}
            else:
                response = {"images": response_body.get("images")}

            return response, height, width, steps
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    """
    Invoke the Bedrock model to generate text from prompts.

    Args:
        body (dict): The request body containing the input prompts.
        model_kwargs (dict, optional): Additional model parameters. Defaults to an empty dict.
        additional_model_fields (dict, optional): Additional model fields. Defaults to an empty dict.

    Returns:
        str: The generated text.

    Raises:
        Exception: If an error occurs during the inference process.
    """
    def invoke_text(self, body, model_kwargs: dict = dict(), additional_model_fields: dict = dict(), tool_config: dict = dict()):
        try:
            provider = self.model_id.split(".")[0]
            is_messages_api = self.messages_api.lower() in ["true"]

            if is_messages_api:
                system = [{"text": model_kwargs["system"]}] if "system" in model_kwargs else list()

                if "system" in model_kwargs:
                    del model_kwargs["system"]

                messages = self._decode_documents(body["inputs"])
                messages = self._decode_images(messages)

                if bool(tool_config):
                    logger.info(f"Using tools {tool_config}")

                    response = self.bedrock_client.converse(
                        modelId=self.model_id,
                        messages=messages,
                        system=system,
                        inferenceConfig=model_kwargs,
                        additionalModelRequestFields=additional_model_fields,
                        toolConfig=tool_config
                    )
                else:
                    response = self.bedrock_client.converse(
                        modelId=self.model_id,
                        messages=messages,
                        system=system,
                        inferenceConfig=model_kwargs,
                        additionalModelRequestFields=additional_model_fields
                    )

                output_message = response['output']['message']

                self.input_tokens = response['usage']['inputTokens']
                self.output_tokens = response['usage']['outputTokens']

                tmp_response = ""
                tmp_tools = []

                for content in output_message['content']:
                    if "text" in content:
                        tmp_response += content['text'] + " "
                    if "toolUse" in content:
                        tmp_tools.append({"toolUse": content["toolUse"]})

                if len(tmp_tools) > 0:
                    response = tmp_tools
                else:
                    response = tmp_response.rstrip()
            else:
                request_body = LLMInputOutputAdapter.prepare_input(
                    provider=provider,
                    prompt=body["inputs"],
                    model_kwargs=model_kwargs
                )

                request_body = json.dumps(request_body)
                model_id = self.model_arn or self.model_id

                response = self.bedrock_client.invoke_model(
                    body=request_body,
                    modelId=model_id,
                    accept="application/json",
                    contentType="application/json"
                )

                response = LLMInputOutputAdapter.prepare_output(provider, response)
                response = response["text"]

                self.input_tokens = get_tokens(body["inputs"])
                self.output_tokens = get_tokens(response)

            return response
        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error(stacktrace)

            raise e