# AWS Deployment Notes

These are the manual steps to wire this pipeline up to AWS so it runs
automatically whenever a new PDF lands in S3. Do this AFTER you've
validated the pipeline locally with `scripts/run_local_demo.py` --
don't debug pipeline logic and AWS plumbing at the same time.

## 1. Create the S3 bucket (raw landing zone)

```
aws s3 mb s3://your-arxiv-pipeline-bucket
```

This reuses the same AWS account as your job-market-analysis project,
so no new account setup is needed.

## 2. Decide where Qdrant lives

Lambda execution environments are ephemeral and have no persistent
local storage, so Qdrant cannot run "inside" your Lambda function --
it needs to be a separately running server that Lambda calls over the
network. Two reasonable options for a portfolio project:

  - **Small EC2 instance** running the same `docker-compose.yml` from
    this repo. Simple, cheap on free tier, and you already understand
    EC2 conceptually if you've used Lambda before.
  - **Qdrant Cloud free tier** -- a managed Qdrant instance, zero infra
    to babysit, but less "infra you stood up yourself" to talk about
    in an interview.

Either way, you'll end up with a URL (e.g. `http://<ec2-ip>:6333` or
your Qdrant Cloud cluster URL) to set as the `QDRANT_URL` environment
variable on your Lambda function in step 5.

If you go the EC2 route, remember to open port 6333 in that instance's
security group ONLY to your Lambda function's outbound traffic /
VPC -- not to the open internet, since Qdrant has no auth by default.

## 3. Build and push the Lambda container image

```
cd arxiv-knowledge-pipeline

aws ecr create-repository --repository-name arxiv-pipeline

aws ecr get-login-password --region <your-region> | \
    docker login --username AWS --password-stdin \
    <account-id>.dkr.ecr.<your-region>.amazonaws.com

docker build -t arxiv-pipeline -f infra/Dockerfile.lambda .

docker tag arxiv-pipeline:latest \
    <account-id>.dkr.ecr.<your-region>.amazonaws.com/arxiv-pipeline:latest

docker push <account-id>.dkr.ecr.<your-region>.amazonaws.com/arxiv-pipeline:latest
```

Expect this image to be large (1-3GB) because of torch +
sentence-transformers + the baked-in models. Lambda supports container
images up to 10GB, so this is fine, but the build/push will take a
few minutes.

## 4. Create the Lambda function

```
aws lambda create-function \
    --function-name arxiv-pipeline \
    --package-type Image \
    --code ImageUri=<account-id>.dkr.ecr.<your-region>.amazonaws.com/arxiv-pipeline:latest \
    --role arn:aws:iam::<account-id>:role/<your-lambda-execution-role> \
    --timeout 300 \
    --memory-size 2048
```

Notes:
  - **Memory**: 2048MB is a reasonable starting point given torch +
    sentence-transformers' footprint. Watch CloudWatch metrics after
    a few runs and tune down if usage is consistently lower (same
    spirit as the 256MB/232MB peak you tuned in your other project).
  - **Timeout**: 300s (5 min) gives headroom for PDF parsing +
    embedding a multi-page paper. Tune based on actual observed
    duration.
  - The execution role needs: `AmazonS3ReadOnlyAccess` (or scoped to
    just your bucket), plus standard Lambda execution permissions
    (`AWSLambdaBasicExecutionRole` for CloudWatch logging).

Set environment variables on the function:

```
aws lambda update-function-configuration \
    --function-name arxiv-pipeline \
    --environment "Variables={QDRANT_URL=http://<your-qdrant-host>:6333,QDRANT_COLLECTION=arxiv_papers}"
```

## 5. Configure the S3 event trigger

This is the key difference from your other project's EventBridge cron
trigger -- this pipeline is event-driven, firing immediately when a
new PDF lands, rather than on a fixed schedule.

```
aws lambda add-permission \
    --function-name arxiv-pipeline \
    --statement-id s3-trigger \
    --action lambda:InvokeFunction \
    --principal s3.amazonaws.com \
    --source-arn arn:aws:s3:::your-arxiv-pipeline-bucket

aws s3api put-bucket-notification-configuration \
    --bucket your-arxiv-pipeline-bucket \
    --notification-configuration '{
        "LambdaFunctionConfigurations": [
            {
                "LambdaFunctionArn": "arn:aws:lambda:<region>:<account-id>:function:arxiv-pipeline",
                "Events": ["s3:ObjectCreated:Put"],
                "Filter": {
                    "Key": {
                        "FilterRules": [
                            {"Name": "suffix", "Value": ".pdf"}
                        ]
                    }
                }
            }
        ]
    }'
```

The `suffix: .pdf` filter is important -- without it, the metadata
sidecar JSON upload (which happens right after the PDF upload in
fetch_arxiv.py) would also trigger the Lambda, which then can't find
the PDF it expects.

## 6. Test it

```
python src/ingest/fetch_arxiv.py \
    --query "retrieval augmented generation" \
    --category cs.CL \
    --max-results 3 \
    --bucket your-arxiv-pipeline-bucket
```

Then check CloudWatch Logs for the `arxiv-pipeline` function to
confirm it fired and processed the uploaded PDFs, and check your
Qdrant collection's point count to confirm chunks landed.

## 7. Monitoring

Same pattern as your other project: CloudWatch for Lambda
duration/memory/error metrics. Worth screenshotting actual
invocation metrics (duration, memory used, cold start time) for your
README/resume bullets once you've run this for real, the same way you
captured "Lambda timeout: 5 minutes... peak usage: 232 MB" for the
job-market-analysis project.
