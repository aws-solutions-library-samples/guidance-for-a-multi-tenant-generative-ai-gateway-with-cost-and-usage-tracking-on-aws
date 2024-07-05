import io
import json
import logging
import math
import traceback

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)
def get_tokens(string):
    logger.info("Counting approximation tokens")

    return math.floor(len(string) / 4)

class SageMakerInferenceStream:
    def __init__(self, sagemaker_runtime, endpoint_name):
        self.sagemaker_runtime = sagemaker_runtime
        self.endpoint_name = endpoint_name
        # A buffered I/O stream to combine the payload parts:
        self.buff = io.BytesIO()
        self.read_pos = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def get_input_tokens(self):
        return self.input_tokens

    def get_output_tokens(self):
        return self.output_tokens

    def invoke_text_streaming(self, body, model_kwargs):
        try:
            request_body = {
                "inputs": body["inputs"],
                "parameters": model_kwargs
            }

            stream = self.stream(request_body)

            response = self.prepare_output_stream_messages_api(stream)

            self.input_tokens = get_tokens(body["inputs"])

            return response

        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error(stacktrace)

            raise e

    def prepare_output_stream_messages_api(self, stream):
        tmp_response = ""
        for part in stream:
            tmp_response += part

        try:
            response = json.loads(tmp_response)
        except json.JSONDecodeError:
            # Invalid JSON, try to fix it
            if not tmp_response.endswith("}"):
                # Missing closing bracket
                tmp_response = tmp_response + "}"
            if not tmp_response.endswith("]"):
                # Uneven brackets
                tmp_response = tmp_response + "]"

            # Try again
            response = json.loads(tmp_response)

        response = response[0]["generated_text"]

        self.output_tokens = get_tokens(response)

        return response

    def stream(self, request_body):
        # Gets a streaming inference response
        # from the specified model endpoint:
        response = self.sagemaker_runtime \
            .invoke_endpoint_with_response_stream(
            EndpointName=self.endpoint_name,
            Body=json.dumps(request_body),
            ContentType="application/json"
        )
        # Gets the EventStream object returned by the SDK:
        event_stream = response['Body']
        for event in event_stream:
            # Passes the contents of each payload part
            # to be concatenated:
            self._write(event['PayloadPart']['Bytes'])
            # Iterates over lines to parse whole JSON objects:
            for line in self._readlines():
                # Returns parts incrementally:
                yield line.decode("utf-8")

    # Writes to the buffer to concatenate the contents of the parts:
    def _write(self, content):
        self.buff.seek(0, io.SEEK_END)
        self.buff.write(content)

    # The JSON objects in buffer end with '\n'.
    # This method reads lines to yield a series of JSON objects:
    def _readlines(self):
        self.buff.seek(self.read_pos)
        for line in self.buff.readlines():
            self.read_pos += len(line)
            yield line[:-1]
