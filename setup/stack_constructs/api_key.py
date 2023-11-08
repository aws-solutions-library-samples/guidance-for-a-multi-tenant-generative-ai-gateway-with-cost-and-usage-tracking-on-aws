from constructs import Construct
from aws_cdk import (
    aws_apigateway as apigw
)

class APIKey(Construct):
    def __init__(
        self,
        scope: Construct,
        id: str,
        prefix:str
    ):
        super().__init__(scope, id)

        self.id = id
        self.prefix = prefix

    def build(
            self,
            rest_api_id: str,
            resource_id: str,
            throttling_rate: int = 10000,
            burst_rate: int = 5000
    ):
        # Lookup RestApi
        api = apigw.RestApi.from_rest_api_attributes(
            self,
            f"{self.id}_rest_api",
            rest_api_id=rest_api_id,
            root_resource_id=resource_id
        )

        # Create API key
        api_key = apigw.ApiKey(
            self,
            f"{self.id}_api_key",
            description=f"API Key for {self.id}",
            enabled=True
        )

        # Create Stage

        stage = apigw.Stage(
            self,
            f"{self.id}_stage",
            deployment=apigw.Deployment(self, f"{self.id}_deployment", api=api),
            metrics_enabled=True,
            throttling_rate_limit=throttling_rate,
            throttling_burst_limit=burst_rate,
            stage_name=f"{self.prefix}_prod"
        )

        # Create Usage Plan
        usage_plan = api.add_usage_plan(
            id=f"{self.id}_usage_plan",
            api_stages=[
                {
                    "api": api,
                    "stage": stage
                }
            ],
            name=f"{self.id}_plan_prod"
        )

        usage_plan.add_api_key(api_key)

        return stage
