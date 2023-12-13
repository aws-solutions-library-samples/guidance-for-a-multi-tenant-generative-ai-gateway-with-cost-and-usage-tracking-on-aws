from constructs import Construct
from aws_cdk import (
    aws_dynamodb as ddb,
    RemovalPolicy
)

class DynamoDB(Construct):
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
        table = ddb.Table(
            self,
            f"{self.id}_streaming_messages",
            partition_key=ddb.Attribute(
                name="request_id",
                type=ddb.AttributeType.STRING
            ),
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY
        )

        for el in self.dependencies:
            table.node.add_dependency(el)

        return table
