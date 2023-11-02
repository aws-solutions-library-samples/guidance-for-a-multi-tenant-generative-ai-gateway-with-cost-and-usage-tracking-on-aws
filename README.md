# Build and track a internal SaaS service with Amazon Bedrock

In this repository, we show you how to build an internal SaaS service to access foundation models with Amazon Bedrock in a multi-tenant architecture. An internal software as a service (SaaS) for foundation models can address governance requirements while providing a simple and consistent interface for the end users. API gateways are a common design pattern that enable consumption of services with standardization and governance. They can provide loose coupling between model consumers and the model endpoint service that gives flexibility to adapt to changing model versions, architectures and invocation methods. 

Multiple tenants within an enterprise could simply reflect to multiple teams or projects accessing LLMs via REST APIs just like other SaaS services. IT teams can add additional governance and controls over this SaaS layer. In this cdk example, we focus specifically on showcasing multiple tenants with different cost centers accessing the service via API gateway. An internal service is responsible to perform usage and cost tracking per tenant and aggregate that cost for reporting. Additionally, the API layer is updated to allow equal usage across all tenants to match the on-demand limits of the Bedrock service. The cdk template provided here deploys all the required resources to the AWS account. 

![IMAGE_DESCRIPTION](images/architecture.png)

The cdk template deploys the following resources : 
1. API gateway 
2. Lambda functions  to list foundation models on Bedrock and invoke models on Bedrock 
3. Lambda function to aggregate usage and cost metering 
4. EventBridge to trigger the metering aggregation on a regular frequency
5. S3 buckets to store the metering output
6. Cloudwatch logs to collect logs from Lambda invocations

Sample notebook in the notebooks folder can be used to invoke Bedrock as either one of the teams/cost_center. API gateway then routes the request to the Bedrock lambda that invokes Bedrock and logs the usage metrics to cloudwatch. EventBridge triggers the metering lambda on a regular frequnecy to aggregate metrics from the cloudwatch logs and generate aggregate usage and cost metrics for the chosen granularity level. The metrics are stored in S3 and can further be visualized with custom reports. 

## Deploy Stack

### Step 1

Edit the value `<PREFIX_ID>` in the file [app.py](./setup/app.py)

### Step 2

Execute the following commands:

```
chmod +x deploy_stach.sh
```

```
./deploy_stach.sh
```