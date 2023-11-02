from aws_cdk import (
    App,
    CfnParameter,
    CfnOutput,
    RemovalPolicy,
    Stack,
    Tags,
    aws_s3
)
from constructs import Construct
import json
from stack_constructs.api import API
from stack_constructs.api_gw import APIGW
from stack_constructs.api_key import APIKey
from stack_constructs.iam import IAM
from stack_constructs.lambda_function import LambdaFunction
from stack_constructs.lambda_layer import LambdaLayer
from stack_constructs.scheduler import LambdaFunctionScheduler
import traceback

def load_configs(filename):
    """
    Loads config from file
    """

    with open(filename, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config

class BedrockAPIStack(Stack):
    def __init__(self, scope: Construct, id: str, prefix_id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # ==================================================
        # ============== STATIC PARAMETERS =================
        # ==================================================
        self.id = id
        self.lambdas_directory = "./../lambdas"
        self.prefix_id = prefix_id

        # ==================================================
        # ================= PARAMETERS =====================
        # ==================================================
        self.bedrock_endpoint_url = CfnParameter(
            self,
            "bedrock_endpoint_url",
            type="String",
            default="https://bedrock-runtime.us-east-1.amazonaws.com"
        ).value_as_string

        self.bedrock_sdk_url = CfnParameter(
            self,
            "bedrock_sdk_url",
            type="String",
            default="https://d2eo22ngex1n9g.cloudfront.net/Documentation/SDK/bedrock-python-sdk.zip"
        ).value_as_string

        self.langchain_requirements = CfnParameter(
            self,
            "langchain_requirements",
            type="String",
            default="aws-lambda-powertools langchain==0.0.309 pydantic PyYaml"
        ).value_as_string

        self.pandas_requirements = CfnParameter(
            self,
            "pandas_requirements",
            type="String",
            default="pandas"
        ).value_as_string

    def build(self):
        # ==================================================
        # ================== IAM ROLE ======================
        # ==================================================
        iam = IAM(
            scope=self,
            id="iam_role_lambda"
        )

        iam_role = iam.build()

        # ==================================================
        # =============== LAMBDA LAYERS ====================
        # ==================================================

        s3_bucket_layer = aws_s3.Bucket(
            self,
            f"{self.prefix_id}_s3_bucket_layer",
            auto_delete_objects=True,
            removal_policy=RemovalPolicy.DESTROY
        )

        lambda_layer = LambdaLayer(
            scope=self,
            id=f"{self.prefix_id}_lambda_layer",
            s3_bucket=s3_bucket_layer.bucket_name,
            role=iam_role.role_name,
        )

        boto3_layer = lambda_layer.build(
            layer_name=f"{self.prefix_id}_boto3_sdk_layer",
            code_dir=f"{self.lambdas_directory}/lambda_layer_url",
            environments={
                "SDK_DOWNLOAD_URL": self.bedrock_sdk_url,
                "S3_BUCKET": s3_bucket_layer.bucket_name
            }
        )

        langchain_layer = lambda_layer.build(
            layer_name=f"{self.prefix_id}_langchain_layer",
            code_dir=f"{self.lambdas_directory}/lambda_layer_requirements",
            environments={
                "REQUIREMENTS": self.langchain_requirements,
                "S3_BUCKET": s3_bucket_layer.bucket_name
            }
        )

        pandas_layer = lambda_layer.build(
            layer_name=f"{self.prefix_id}_pandas_layer",
            code_dir=f"{self.lambdas_directory}/lambda_layer_requirements",
            environments={
                "REQUIREMENTS": self.pandas_requirements,
                "S3_BUCKET": s3_bucket_layer.bucket_name
            }
        )

        # ==================================================
        # ============= Bedrock Functions ==================
        # ==================================================

        lambda_function = LambdaFunction(
            scope=self,
            id=f"{self.prefix_id}_lambda_function",
            role=iam_role.role_name,
        )

        bedrock_invoke_model = lambda_function.build(
            function_name=f"{self.prefix_id}_bedrock_invoke_model",
            code_dir=f"{self.lambdas_directory}/invoke_model",
            memory=512,
            timeout=900,
            environment={
                "BEDROCK_URL": "https://bedrock-runtime.us-east-1.amazonaws.com",
            },
            layers=[boto3_layer, langchain_layer]
        )

        bedrock_list_model = lambda_function.build(
            function_name=f"{self.prefix_id}_bedrock_list_foundation_models",
            code_dir=f"{self.lambdas_directory}/list_foundation_models",
            memory=512,
            timeout=900,
            environment={
                "BEDROCK_URL": self.bedrock_endpoint_url
            },
            layers=[boto3_layer]
        )

        # ==================================================
        # =============== Lambda Metering ==================
        # ==================================================

        s3_bucket_metering = aws_s3.Bucket(
            self,
            f"{self.prefix_id}_s3_bucket_metering",
            bucket_name=f"{self.prefix_id}-bucket-metering-bedrock",
            auto_delete_objects=True,
            removal_policy=RemovalPolicy.DESTROY
        )

        bedrock_metering = lambda_function.build(
            function_name=f"{self.prefix_id}_bedrock_metering",
            code_dir=f"{self.lambdas_directory}/metering",
            memory=512,
            timeout=900,
            environment={
                "LOG_GROUP_API": f"/aws/lambda/{self.prefix_id}_bedrock_invoke_model",
                "S3_BUCKET": s3_bucket_metering.bucket_name
            },
            layers=[pandas_layer]
        )

        scheduler = LambdaFunctionScheduler(
            self,
            id=f"{self.prefix_id}_lambda_scheduler"
        )

        scheduler.build(
            lambda_function=bedrock_metering
        )

        # ==================================================
        # ================== API Gateway ===================
        # ==================================================

        api_gw_class = APIGW(
            self,
            id=f"{self.prefix_id}_api_gw",
            api_gw_name=f"{self.prefix_id}_bedrock_api_gw"
        )

        api_gw = api_gw_class.build()

        # ==================================================
        # =================== API Key ======================
        # ==================================================

        api_key_class = APIKey(
            self,
            id=f"{self.prefix_id}_api_key",
        )

        api_key_class.build(
            api=api_gw
        )

        # ==================================================
        # ================== API Routes ====================
        # ==================================================

        api_route = API(
            self,
            id=f"{self.prefix_id}_api_route",
            api_gw=api_gw,

        )

        api_route.build(
            lambda_function=bedrock_invoke_model,
            route="invoke_model",
            method="POST"
        )

        api_route.build(
            lambda_function=bedrock_list_model,
            route="list_foundation_models",
            method="GET"
        )

        CfnOutput(self, f"{self.prefix_id}_api_gw_url", export_name=f"{self.prefix_id}ApiGatewayUrl", value=api_gw.url)

# ==================================================
# ============== STACK WITH COST CENTER ============
# ==================================================

app = App()

configs = load_configs("./configs.json")

for config in configs:
    api_stack = BedrockAPIStack(
        scope=app, id=f"{config['STACK_PREFIX']}-bedrock-saas", prefix_id=config['STACK_PREFIX']
    )

    api_stack.build()

    # Add a cost tag to all constructs in the stack
    Tags.of(api_stack).add("Tenant", api_stack.prefix_id)

try:
    app.synth()
except Exception as e:
    stacktrace = traceback.format_exc()
    print(stacktrace)

    raise e
