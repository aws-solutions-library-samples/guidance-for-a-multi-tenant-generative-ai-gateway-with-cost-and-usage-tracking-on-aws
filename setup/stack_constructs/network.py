from constructs import Construct
from aws_cdk import (
    aws_ec2 as ec2
)

class Network(Construct):
    def __init__(
        self,
        scope: Construct,
        id: str,
        account: str,
        region: str,
        dependencies: list = []
    ):
        super().__init__(scope, id)

        self.id = id
        self.account = account
        self.region = region
        self.dependencies = dependencies

    def build(
            self,
            vpc_cidr: str
    ):
        # Resources
        vpc = ec2.Vpc(
            self,
            f"{self.id}_vpc",
            ip_addresses=ec2.IpAddresses.cidr(vpc_cidr),
            enable_dns_hostnames=True,
            enable_dns_support=True,
            gateway_endpoints={
                "S3": ec2.GatewayVpcEndpointOptions(service=ec2.GatewayVpcEndpointAwsService.S3),
                "DynamoDB": ec2.GatewayVpcEndpointOptions(service=ec2.GatewayVpcEndpointAwsService.DYNAMODB)
            },
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name=f"{self.id}_private_subnet_1",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name=f"{self.id}_private_subnet_2",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24
                )
            ]
        )

        # Lookup private subnets
        private_subnet1 = ec2.Subnet.from_subnet_attributes(
            self, f"{self.id}_private_subnet_1",
            subnet_id=vpc.isolated_subnets[0].subnet_id
        )

        private_subnet2 = ec2.Subnet.from_subnet_attributes(
            self, f"{self.id}_private_subnet_2",
            subnet_id=vpc.isolated_subnets[1].subnet_id
        )

        security_group = ec2.SecurityGroup(
            self,
            f"{self.id}_security_group",
            vpc=vpc,
            allow_all_outbound=True,
            description="security group for bedrock workload in private subnets",
        )

        endpoint_security_group = ec2.SecurityGroup(
            self,
            f"{self.id}_vpce_security_group",
            vpc=vpc,
            description="Allow TLS for VPC Endpoint",
        )

        endpoint_security_group.add_ingress_rule(
            peer=security_group,
            connection=ec2.Port.tcp(443)
        )

        # Bedrock VPCE
        ec2.CfnVPCEndpoint(
            self,
            f"{self.id}_vpce_bedrock",
            service_name=f"com.amazonaws.{self.region}.bedrock",
            vpc_id=vpc.vpc_id,
            private_dns_enabled=True,
            security_group_ids=[endpoint_security_group.security_group_id],
            subnet_ids=[private_subnet1.subnet_id, private_subnet2.subnet_id],
            vpc_endpoint_type="Interface"
        )

        # Bedrock Runtime VPCE
        ec2.CfnVPCEndpoint(
            self,
            f"{self.id}_vpce_bedrock_runtime",
           service_name=f"com.amazonaws.{self.region}.bedrock-runtime",
           vpc_id=vpc.vpc_id,
           private_dns_enabled=True,
           security_group_ids=[endpoint_security_group.security_group_id],
           subnet_ids=[private_subnet1.subnet_id, private_subnet2.subnet_id],
           vpc_endpoint_type="Interface"
        )

        # API Gateway VPCE
        ec2.CfnVPCEndpoint(
            self,
            f"{self.id}_vpce_api_gw",
            service_name=f"com.amazonaws.{self.region}.execute-api",
            vpc_id=vpc.vpc_id,
            private_dns_enabled=True,
            security_group_ids=[endpoint_security_group.security_group_id],
            subnet_ids=[private_subnet1.subnet_id, private_subnet2.subnet_id],
            vpc_endpoint_type="Interface"
        )

        # CloudWatch Logs VPCE
        ec2.CfnVPCEndpoint(
            self,
            f"{self.id}_vpce_logs",
            service_name=f"com.amazonaws.{self.region}.logs",
            vpc_id=vpc.vpc_id,
            private_dns_enabled=True,
            security_group_ids=[endpoint_security_group.security_group_id],
            subnet_ids=[private_subnet1.subnet_id, private_subnet2.subnet_id],
            vpc_endpoint_type="Interface"
        )

        # Events VPCE
        ec2.CfnVPCEndpoint(
            self,
            f"{self.id}_vpce_events",
            service_name=f"com.amazonaws.{self.region}.events",
            vpc_id=vpc.vpc_id,
            private_dns_enabled=True,
            security_group_ids=[endpoint_security_group.security_group_id],
            subnet_ids=[private_subnet1.subnet_id, private_subnet2.subnet_id],
            vpc_endpoint_type="Interface"
        )

        # Lambda VPCE
        ec2.CfnVPCEndpoint(
            self,
            f"{self.id}_vpce_lambda",
            service_name=f"com.amazonaws.{self.region}.lambda",
            vpc_id=vpc.vpc_id,
            private_dns_enabled=True,
            security_group_ids=[endpoint_security_group.security_group_id],
            subnet_ids=[private_subnet1.subnet_id, private_subnet2.subnet_id],
            vpc_endpoint_type="Interface"
        )

        # SageMaker API VPCE
        ec2.CfnVPCEndpoint(
            self,
            f"{self.id}_vpce_sagemaker_api",
            service_name=f"com.amazonaws.{self.region}.sagemaker.api",
            vpc_id=vpc.vpc_id,
            private_dns_enabled=True,
            security_group_ids=[endpoint_security_group.security_group_id],
            subnet_ids=[private_subnet1.subnet_id, private_subnet2.subnet_id],
            vpc_endpoint_type="Interface"
        )

        # SageMaker Runtime VPCE
        ec2.CfnVPCEndpoint(
            self,
            f"{self.id}_vpce_sagemaker_runtime",
            service_name=f"com.amazonaws.{self.region}.sagemaker.runtime",
            vpc_id=vpc.vpc_id,
            private_dns_enabled=True,
            security_group_ids=[endpoint_security_group.security_group_id],
            subnet_ids=[private_subnet1.subnet_id, private_subnet2.subnet_id],
            vpc_endpoint_type="Interface"
        )

        # SageMaker Metrics VPCE
        ec2.CfnVPCEndpoint(
            self,
            f"{self.id}_vpce_sagemaker_metrics",
            service_name=f"com.amazonaws.{self.region}.sagemaker.metrics",
            vpc_id=vpc.vpc_id,
            private_dns_enabled=True,
            security_group_ids=[endpoint_security_group.security_group_id],
            subnet_ids=[private_subnet1.subnet_id, private_subnet2.subnet_id],
            vpc_endpoint_type="Interface"
        )

        # SageMaker Runtime FIPS VPCE
        ec2.CfnVPCEndpoint(
            self,
            f"{self.id}_vpce_sagemaker_runtime_fips",
            service_name=f"com.amazonaws.{self.region}.sagemaker.runtime-fips",
            vpc_id=vpc.vpc_id,
            private_dns_enabled=True,
            security_group_ids=[endpoint_security_group.security_group_id],
            subnet_ids=[private_subnet1.subnet_id, private_subnet2.subnet_id],
            vpc_endpoint_type="Interface"
        )

        for el in self.dependencies:
            vpc.node.add_dependency(el)

        return vpc, private_subnet1, private_subnet2, security_group
