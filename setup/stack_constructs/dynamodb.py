from constructs import Construct
from aws_cdk import (
    aws_dynamodb as ddb,
)

class DynamoDB(Construct):
    def __init__(
        self,
        scope: Construct,
        id: str
    ):
        super().__init__(scope, id)

        self.id = id

    def build(self):
        table = ddb.Table(
            self,
            f"{self.id}_streaming_messages",
            partition_key=ddb.Attribute(
                name="request_id",
                type=ddb.AttributeType.STRING
            ),
            time_to_live_attribute="ttl"
        )

        return table
