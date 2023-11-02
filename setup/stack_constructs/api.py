from constructs import Construct
from aws_cdk import (
    aws_apigateway as apigw,
    aws_iam as iam,
    aws_lambda as lambda_
)

class API(Construct):
    def __init__(
        self,
        scope: Construct,
        id: str,
        api_gw: apigw.LambdaRestApi,
    ):
        super().__init__(scope, id)

        self.id = id
        self.api_gw = api_gw

    def build(
            self,
            lambda_function: lambda_.Function,
            route: str,
            method: str
    ):
        # Add method/route

        lambda_function.add_permission(
            id=f"{self.id}_{route}_permission",
            action="lambda:InvokeFunction",
            principal=iam.ServicePrincipal("apigateway.amazonaws.com"),
            source_arn=self.api_gw.arn_for_execute_api(
                stage="*",
                method=method,
                path=f"/{route}"
            )
        )

        resourse = self.api_gw.root.add_resource(route)
        resourse.add_method(
            http_method=method,
            integration=apigw.LambdaIntegration(lambda_function),
            api_key_required=True,
            method_responses=[
                apigw.MethodResponse(
                    status_code="401",
                    response_parameters={
                        "method.response.header.Access-Control-Allow-Origin": True,
                    },
                    response_models={
                        "application/json": apigw.Model.ERROR_MODEL,
                    }
                )
            ]
        )
