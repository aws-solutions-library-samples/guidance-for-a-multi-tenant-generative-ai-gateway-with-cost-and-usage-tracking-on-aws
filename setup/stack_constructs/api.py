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
        dependencies: list = []
    ):
        super().__init__(scope, id)

        self.id = id
        self.api_gw = api_gw
        self.dependencies = dependencies

    def build(
        self,
        lambda_function: lambda_.Function,
        route: str,
        method: str,
        validator: bool = False
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

        if validator:
            resourse.add_method(
                http_method=method,
                integration=apigw.LambdaIntegration(lambda_function),
                api_key_required=True,
                request_parameters={
                    "method.request.header.team_id": True,
                    "method.request.header.streaming": False,
                    "method.request.header.type": False,
                    "method.request.querystring.model_id": True,
                    "method.request.querystring.requestId": False
                },
                request_validator_options={
                    "request_validator_name": "parameter-validator",
                    "validate_request_parameters": True,
                    "validate_request_body": False
                },
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
        else:
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

        return resourse
