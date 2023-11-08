from constructs import Construct
from aws_cdk import (
    aws_apigateway as apigw,
    Duration
)

class APIGW(Construct):
    def __init__(
        self,
        scope: Construct,
        id: str,
        api_gw_name: str,
    ):
        super().__init__(scope, id)

        self.id = id
        self.api_gw_name = api_gw_name

    def build(
            self
    ):
        # Create API Gateway REST
        api = apigw.RestApi(
            scope=self,
            id=f"{self.id}_api_gateway",
            rest_api_name=self.api_gw_name,
            deploy=False
        )

        api.timeout = Duration.seconds(300)

        return api
