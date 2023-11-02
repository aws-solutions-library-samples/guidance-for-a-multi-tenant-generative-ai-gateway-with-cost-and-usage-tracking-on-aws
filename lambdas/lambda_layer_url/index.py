import os
import sys
import re
import shutil
import subprocess
import boto3
import zipfile
import urllib3
from datetime import datetime
import cfnresponse

bedrock_sdk_url = os.environ['SDK_DOWNLOAD_URL']
s3_bucket = os.environ['S3_BUCKET']


def download_file_from_url(url, local_path):
    """Download a file from a URL to a local save path."""
    http = urllib3.PoolManager()
    response = http.request('GET', url)
    if response.status == 200:
        with open(local_path, 'wb') as file:
            file.write(response.data)
        print("File downloaded successfully.")
    else:
        print("Failed to download the file.", response)


def upload_file_to_s3(file_path, bucket, key):
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket, key)
    print(f"Upload successful. {file_path} uploaded to {bucket}/{key}")


def extract_file_from_zip(zip_file_path, file_name):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extract(file_name)
        print(f"Successfully extracted {file_name} from {zip_file_path}")


def find_boto_wheels(zipname):
    zipf = zipfile.ZipFile(zipname, 'r')
    zip_files = zipf.namelist()
    b = re.compile('boto3(.*)\.whl')
    bc = re.compile('botocore(.*)\.whl')
    boto3_whl_file = [s for s in zip_files if b.match(s)][0]
    botocore_whl_file = [s for s in zip_files if bc.match(s)][0]

    return boto3_whl_file, botocore_whl_file


def make_zip_filename():
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    filename = f'BedrockBoto3SDK_{timestamp}.zip'
    return filename


def zipdir(path, zipname):
    zipf = zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(path):
        for file in files:
            zipf.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))
    zipf.close()


def empty_bucket(bucket_name):
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    if 'Contents' in response:
        keys = [{'Key': obj['Key']} for obj in response['Contents']]
        s3_client.delete_objects(Bucket=bucket_name, Delete={'Objects': keys})
    return


def lambda_handler(event, context):
    print("Event: ", event)
    responseData = {}
    reason = ""
    status = cfnresponse.SUCCESS
    try:
        if event['RequestType'] != 'Delete':
            os.chdir('/tmp')
            # download Bedrock SDK
            zip_file_name = 'bedrock-python-sdk.zip'
            print(f"downloading from {bedrock_sdk_url} to {zip_file_name}")
            download_file_from_url(bedrock_sdk_url, zip_file_name)
            boto3_whl_file, botocore_whl_file = find_boto_wheels(zip_file_name)
            extract_file_from_zip(zip_file_name, botocore_whl_file)
            extract_file_from_zip(zip_file_name, boto3_whl_file)
            if os.path.exists("python"):
                shutil.rmtree("python")
            os.mkdir("python")
            print(f"running pip install botocore")
            subprocess.check_call([sys.executable, "-m", "pip", "install", botocore_whl_file, "-t", "python"])
            print(f"running pip install boto3")
            subprocess.check_call([sys.executable, "-m", "pip", "install", boto3_whl_file, "-t", "python"])
            boto3_zip_name = make_zip_filename()
            zipdir("python", boto3_zip_name)
            print(f"uploading {boto3_zip_name} to s3 bucket {s3_bucket}")
            upload_file_to_s3(boto3_zip_name, s3_bucket, boto3_zip_name)
            responseData = {"Bucket": s3_bucket, "Key": boto3_zip_name}
        else:
            # delete - empty the bucket so it can be deleted by the stack.
            empty_bucket(s3_bucket)
    except Exception as e:
        print(e)
        status = cfnresponse.FAILED
        reason = f"Exception thrown: {e}"
    cfnresponse.send(event, context, status, responseData, reason=reason)