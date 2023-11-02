from constructs import Construct
from aws_cdk import (
    App,
    CfnOutput,
    RemovalPolicy,
    Stack,
    Tags,
    aws_s3
)

from stack_constructs.api import API
from stack_constructs.api_gw import APIGW
from stack_constructs.api_key import APIKey
from stack_constructs.iam import IAM
from stack_constructs.lambda_function import LambdaFunction
from stack_constructs.lambda_layer import LambdaLayer
from stack_constructs.scheduler import LambdaFunctionScheduler
import traceback

class BedrockAPIStack(Stack):
    def __init__(self, scope: Construct, id: str, prefix_id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # ==================================================
        # ================= PARAMETERS =====================
        # ==================================================
        bedrock_endpoint_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
        bedrock_sdk_url = "https://d2eo22ngex1n9g.cloudfront.net/Documentation/SDK/bedrock-python-sdk.zip"
        lambdas_directory = "./../lambdas"
        langchain_requirements = "aws-lambda-powertools langchain==0.0.309 pydantic PyYaml"
        pandas_requirements = "pandas"

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
            f"{prefix_id}_s3_bucket_layer",
            auto_delete_objects=True,
            removal_policy=RemovalPolicy.DESTROY
        )

        lambda_layer = LambdaLayer(
            scope=self,
            id=f"{prefix_id}_lambda_layer",
            s3_bucket=s3_bucket_layer.bucket_name,
            role=iam_role.role_name,
        )

        boto3_layer = lambda_layer.build(
            layer_name=f"{prefix_id}_boto3_sdk_layer",
            code_dir=f"{lambdas_directory}/lambda_layer_url",
            environments={
                "SDK_DOWNLOAD_URL": bedrock_sdk_url,
                "S3_BUCKET": s3_bucket_layer.bucket_name
            }
        )

        langchain_layer = lambda_layer.build(
            layer_name=f"{prefix_id}_langchain_layer",
            code_dir=f"{lambdas_directory}/lambda_layer_requirements",
            environments={
                "REQUIREMENTS": langchain_requirements,
                "S3_BUCKET": s3_bucket_layer.bucket_name
            }
        )

        pandas_layer = lambda_layer.build(
            layer_name=f"{prefix_id}_pandas_layer",
            code_dir=f"{lambdas_directory}/lambda_layer_requirements",
            environments={
                "REQUIREMENTS": pandas_requirements,
                "S3_BUCKET": s3_bucket_layer.bucket_name
            }
        )

        # ==================================================
        # ============= Bedrock Functions ==================
        # ==================================================

        lambda_function = LambdaFunction(
            scope=self,
            id=f"{prefix_id}_lambda_function",
            role=iam_role.role_name,
        )

        bedrock_invoke_model = lambda_function.build(
            function_name=f"{prefix_id}_bedrock_invoke_model",
            code_dir=f"{lambdas_directory}/invoke_model",
            memory=512,
            timeout=900,
            environment={
                "BEDROCK_URL": "https://bedrock-runtime.us-east-1.amazonaws.com",
            },
            layers=[boto3_layer, langchain_layer]
        )

        bedrock_list_model = lambda_function.build(
            function_name=f"{prefix_id}_bedrock_list_foundation_models",
            code_dir=f"{lambdas_directory}/list_foundation_models",
            memory=512,
            timeout=900,
            environment={
                "BEDROCK_URL": bedrock_endpoint_url
            },
            layers=[boto3_layer]
        )

        # ==================================================
        # =============== Lambda Metering ==================
        # ==================================================

        s3_bucket_metering = aws_s3.Bucket(
            self,
            f"{prefix_id}_s3_bucket_metering",
            bucket_name=f"{prefix_id}-bucket-metering-bedrock",
            auto_delete_objects=True,
            removal_policy=RemovalPolicy.DESTROY
        )

        bedrock_metering = lambda_function.build(
            function_name=f"{prefix_id}_bedrock_metering",
            code_dir=f"{lambdas_directory}/metering",
            memory=512,
            timeout=900,
            environment={
                "LOG_GROUP_API": f"/aws/lambda/{prefix_id}_bedrock_invoke_model",
                "S3_BUCKET": s3_bucket_metering.bucket_name
            },
            layers=[pandas_layer]
        )

        scheduler = LambdaFunctionScheduler(
            self,
            id=f"{prefix_id}_lambda_scheduler"
        )

        scheduler.build(
            lambda_function=bedrock_metering
        )

        # ==================================================
        # ================== API Gateway ===================
        # ==================================================

        api_gw_class = APIGW(
            self,
            id=f"{prefix_id}_api_gw",
            api_gw_name=f"{prefix_id}_bedrock_api_gw"
        )

        api_gw = api_gw_class.build()

        # ==================================================
        # =================== API Key ======================
        # ==================================================

        api_key_class = APIKey(
            self,
            id=f"{prefix_id}_api_key",
        )

        api_key_class.build(
            api=api_gw
        )

        # ==================================================
        # ================== API Routes ====================
        # ==================================================

        api_route = API(
            self,
            id=f"{prefix_id}_api_route",
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

        CfnOutput(self, f"{prefix_id}_api_gw_url", export_name=f"{prefix_id}ApiGatewayUrl", value=api_gw.url)

# ==================================================
# ============== STACK WITH COST CENTER ============
# ==================================================
prefix_id = "<PREFIX_ID>"

app = App()
api_stack = BedrockAPIStack(
    scope=app, id=f"{prefix_id}-Bedrock-SaaS", prefix_id=prefix_id
)
# Add a cost tag to all constructs in the stack
Tags.of(api_stack).add("CostCenter", prefix_id)

try:
    app.synth()
except Exception as e:
    stacktrace = traceback.format_exc()
    print(stacktrace)

    raise e
