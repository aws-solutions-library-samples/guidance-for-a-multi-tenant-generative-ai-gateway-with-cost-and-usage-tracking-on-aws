from constructs import Construct
from aws_cdk import (
    aws_apigateway as apigw,
)

class APIKey(Construct):
    def __init__(
        self,
        scope: Construct,
        id: str
    ):
        super().__init__(scope, id)

        self.id = id

    def build(
            self,
            api: apigw.RestApi
    ):

        # Create API key
        api_key = apigw.ApiKey(
            self,
            f"{self.id}_api_key",
            description=f"API Key for {self.id}",
            enabled=True
        )

        # Create Deployment
        deployment = apigw.Deployment(
            self,
            f"{self.id}_deployment",
            api=api
        )

        stage = apigw.Stage(
            self, f"{self.id}_stage_prod",
            deployment=deployment,
            stage_name="prod"
        )

        # Associate API key with Usage Plan
        usage_plan = api.add_usage_plan(
            id=f"{self.id}_usage_plan",
            api_stages=[
                {
                    "api": api,
                    "stage": stage
                }
            ],
            name=f"{self.id}_usage_plan_prod"
        )

        usage_plan.add_api_key(api_key)
