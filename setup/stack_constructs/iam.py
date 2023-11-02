from constructs import Construct
from aws_cdk import (
    aws_iam as iam
)

class IAM(Construct):
    def __init__(
            self,
            scope: Construct,
            id: str,
    ):
        super().__init__(scope, id)

        self.id = id

    def build(self):
        # ==================================================
        # ================= IAM ROLE =======================
        # ==================================================
        self.lambda_role = iam.Role(
            self,
            id=f"{self.id}_role",
            assumed_by=iam.ServicePrincipal(service="lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSLambdaExecute"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchLogsFullAccess")
            ],
        )

        self.s3_policy = iam.Policy(
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

        self.bedrock_policy = iam.Policy(
            scope=self,
            id=f"{self.id}_policy_bedrock",
            policy_name="BedrockPolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "sts:AssumeRole",
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

        self.s3_policy.attach_to_role(self.lambda_role)
        self.bedrock_policy.attach_to_role(self.lambda_role)

        return self.lambda_role
