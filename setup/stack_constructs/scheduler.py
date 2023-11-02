from constructs import Construct
from aws_cdk import (
    aws_events as events,
    aws_events_targets as targets,
    aws_lambda as lambda_
)


class LambdaFunctionScheduler(Construct):
    def __init__(
        self,
        scope: Construct,
        id: str,
    ):
        super().__init__(scope, id)

        self.id = id

    def build(
            self,
            lambda_function: lambda_.Function,
    ):
        # ==================================================
        # ================== SCHEDULING ====================
        # ==================================================
        self.cron_rule = events.Rule(
            scope=self,
            id=f"{self.id}_cron_rule",
            rule_name=f"{self.id}_usage_aggregator_schedule",
            schedule=events.Schedule.expression('cron(0 0 * * ? *)')
        )

        self.cron_rule.add_target(target=targets.LambdaFunction(lambda_function))
