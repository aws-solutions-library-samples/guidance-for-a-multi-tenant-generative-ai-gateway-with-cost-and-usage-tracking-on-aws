from aws_cdk import (
    App,
    CfnOutput,
    Fn,
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
from stack_constructs.network import Network
from stack_constructs.scheduler import LambdaFunctionScheduler
import traceback

def _load_configs(filename):
    """
    Loads config from file
    """

    with open(filename, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config

class BedrockAPIStack(Stack):
    def __init__(
            self, scope:
            Construct, id: str,
            config: dict,
            **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        # ==================================================
        # ============== STATIC PARAMETERS =================
        # ==================================================
        self.id = id
        self.lambdas_directory = "./../lambdas"
        self.prefix_id = config.get("STACK_PREFIX", None)
        self.vpc_cidr = config.get("VPC_CIDR", None)

        # ==================================================
        # ================= PARAMETERS =====================
        # ==================================================
        self.bedrock_endpoint_url = config.get("BEDROCK_ENDPOINT", None)
        if self.bedrock_endpoint_url is not None:
            self.bedrock_endpoint_url = self.bedrock_endpoint_url.format(self.region)
        self.bedrock_sdk_url = config.get("BEDROCK_SDK_URL", None)
        self.langchain_requirements = config.get("LANGCHAIN_REQUIREMENTS", None)
        self.pandas_requirements = config.get("PANDAS_REQUIREMENTS", None)
        self.api_throttling_rate = config.get("API_THROTTLING_RATE", 10000)
        self.api_burst_rate = config.get("API_BURST_RATE", 10000)
        self.api_gw_id = config.get("API_GATEWAY_ID", None)
        self.api_gw_resource_id = config.get("API_GATEWAY_RESOURCE_ID", None)

        if self.prefix_id is None:
            raise Exception("STACK_PREFIX not defined")

        if self.vpc_cidr is not None and self.bedrock_endpoint_url is not None and self.bedrock_sdk_url is not None and self.langchain_requirements is not None and self.pandas_requirements is not None:
            self.full_deployment = True
        else:
            if self.api_gw_id is not None and self.api_gw_resource_id is not None:
                self.full_deployment = False
            else:
                raise Exception("API_GATEWAY_ID and API_GATEWAY_RESOURCE_ID not defined")

    def build_full(self):
        # ==================================================
        # ================== IAM ROLE ======================
        # ==================================================
        iam = IAM(
            scope=self,
            id="iam_role_lambda"
        )

        iam_role = iam.build()

        # ==================================================
        # =================== NETWORK ======================
        # ==================================================

        network_class = Network(
            scope=self,
            id="network_stack",
            account=self.account,
            region=self.region
        )

        vpc, private_subnet1, private_subnet2, security_group = network_class.build(
            vpc_cidr=self.vpc_cidr
        )

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
        # ============= BEDROCK FUNCTIONS ==================
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
                "BEDROCK_URL": self.bedrock_endpoint_url,
                "BEDROCK_REGION": self.region
            },
            vpc=vpc,
            subnets=[private_subnet1, private_subnet2],
            security_groups=[security_group],
            layers=[boto3_layer, langchain_layer]
        )

        bedrock_list_model = lambda_function.build(
            function_name=f"{self.prefix_id}_bedrock_list_foundation_models",
            code_dir=f"{self.lambdas_directory}/list_foundation_models",
            memory=512,
            timeout=900,
            environment={
                "BEDROCK_URL": self.bedrock_endpoint_url,
                "BEDROCK_REGION": self.region
            },
            vpc=vpc,
            subnets=[private_subnet1, private_subnet2],
            security_groups=[security_group],
            layers=[boto3_layer]
        )

        # ==================================================
        # =============== LAMBDA METERING ==================
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
            vpc=vpc,
            subnets=[private_subnet1, private_subnet2],
            security_groups=[security_group],
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
        # ================== API GATEWAY ===================
        # ==================================================

        api_gw_class = APIGW(
            self,
            id=f"{self.prefix_id}_api_gw",
            api_gw_name=f"{self.prefix_id}_bedrock_api_gw"
        )

        api_gw = api_gw_class.build()

        # ==================================================
        # =================== API KEY ======================
        # ==================================================

        api_key_class = APIKey(
            self,
            id=f"{self.prefix_id}_api_key",
            prefix=self.prefix_id
        )

        stage = api_key_class.build(
            rest_api_id=api_gw.rest_api_id,
            resource_id=api_gw.rest_api_root_resource_id,
            throttling_rate=self.api_throttling_rate,
            burst_rate=self.api_burst_rate
        )

        # ==================================================
        # ================== API ROUTES ====================
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

        CfnOutput(self, f"{self.prefix_id}_api_gw_url", export_name=f"{self.prefix_id}ApiGatewayUrl", value=stage.url_for_path(path=None))
        CfnOutput(self, f"{self.prefix_id}_api_gw_id", export_name=f"{self.prefix_id}ApiGatewayId", value=api_gw.rest_api_id)
        CfnOutput(self, f"{self.prefix_id}_api_gw_resource_id", export_name=f"{self.prefix_id}ApiGatewayResourceId", value=api_gw.rest_api_root_resource_id)

    def build_api_key(self):
        # ==================================================
        # =================== API KEY ======================
        # ==================================================

        api_key_class = APIKey(
            self,
            id=f"{self.prefix_id}_api_key",
            prefix=self.prefix_id
        )

        stage = api_key_class.build(
            rest_api_id=self.api_gw_id,
            resource_id=self.api_gw_resource_id,
            throttling_rate=self.api_throttling_rate,
            burst_rate=self.api_burst_rate
        )

        CfnOutput(self, f"{self.prefix_id}_api_gw_url", export_name=f"{self.prefix_id}ApiGatewayUrl", value=stage.url_for_path(path=None))

# ==================================================
# ============== STACK WITH COST CENTER ============
# ==================================================

app = App()

configs = _load_configs("./configs.json")

for config in configs:
    api_stack = BedrockAPIStack(
        scope=app,
        id=f"{config['STACK_PREFIX']}-bedrock-saas",
        config=config
    )

    if api_stack.full_deployment:
        api_stack.build_full()
    else:
        api_stack.build_api_key()

    # Add a cost tag to all constructs in the stack
    Tags.of(api_stack).add("Tenant", api_stack.prefix_id)

try:
    app.synth()
except Exception as e:
    stacktrace = traceback.format_exc()
    print(stacktrace)

    raise e
