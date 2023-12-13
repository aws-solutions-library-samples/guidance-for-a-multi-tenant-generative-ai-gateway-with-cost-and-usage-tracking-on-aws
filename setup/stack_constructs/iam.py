from constructs import Construct
from aws_cdk import (
    aws_iam as iam
)

class IAM(Construct):
    def __init__(
            self,
            scope: Construct,
            id: str,
            dependencies: list = []
    ):
        super().__init__(scope, id)

        self.id = id
        self.dependencies = dependencies

    def build(self):
        # ==================================================
        # ================= IAM ROLE =======================
        # ==================================================
        lambda_role = iam.Role(
            self,
            id=f"{self.id}_role",
            assumed_by=iam.ServicePrincipal(service="lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSLambdaExecute"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchLogsFullAccess")
            ],
        )

        ec2_policy = iam.Policy(
            scope=self,
            id=f"{self.id}_policy_ec2",
            policy_name="EC2Policy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        'ec2:AssignPrivateIpAddresses',
                        'ec2:CreateNetworkInterface',
                        'ec2:DeleteNetworkInterface',
                        'ec2:DescribeNetworkInterfaces',
                        'ec2:DescribeSecurityGroups',
                        'ec2:DescribeSubnets',
                        'ec2:DescribeVpcs',
                        'ec2:UnassignPrivateIpAddresses',
                        'ec2:*VpcEndpoint*'
                    ],
                    resources=["*"],
                )
            ],
        )

        lambda_policy = iam.Policy(
            scope=self,
            id=f"{self.id}_policy_lambda",
            policy_name="LambdaPolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        'lambda:InvokeFunction'
                    ],
                    resources=["*"],
                )
            ],
        )

        s3_policy = iam.Policy(
            scope=self,
            id=f"{self.id}_policy_s3",
            policy_name="S3Policy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        's3:PutObject',
                        's3:DeleteObject',
                        's3:ListBucket'
                    ],
                    resources=["*"],
                )
            ],
        )

        dynamodb_policy = iam.Policy(
            scope=self,
            id=f"{self.id}_policy_dynamodb",
            policy_name="DynamoDBPolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "dynamodb:BatchGetItem",
                        "dynamodb:DeleteItem",
                        "dynamodb:GetItem",
                        "dynamodb:PutItem"

                    ],
                    resources=["*"],
                )
            ],
        )

        bedrock_policy = iam.Policy(
            scope=self,
            id=f"{self.id}_policy_bedrock",
            policy_name="BedrockPolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "sts:AssumeRole"
                    ],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "bedrock:*",
                    ],
                    resources=["*"],
                )
            ],
        )

        bedrock_policy.attach_to_role(lambda_role)
        dynamodb_policy.attach_to_role(lambda_role)
        ec2_policy.attach_to_role(lambda_role)
        lambda_policy.attach_to_role(lambda_role)
        s3_policy.attach_to_role(lambda_role)

        for el in self.dependencies:
            lambda_role.node.add_dependency(el)

        return lambda_role
