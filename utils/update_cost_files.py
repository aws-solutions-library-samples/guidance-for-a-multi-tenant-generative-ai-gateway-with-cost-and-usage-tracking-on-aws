import boto3
import pandas as pd
import io

# Initialize the S3 client
s3 = boto3.client('s3')

# Specify your bucket name
bucket_name = '<S3_BUCKET>'
folder_name = 'succeed'

def update():

    # List all objects in the specified folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name + '/')

    # Iterate through each object in the folder
    for obj in response['Contents']:
        # Get the object key (file name)
        key = obj['Key']

        # Check if the file is a CSV
        if key.endswith('.csv'):
            print(f"Processing file: {key}")

            # Download the CSV file
            response = s3.get_object(Bucket=bucket_name, Key=key)
            csv_content = response['Body'].read()

            # Load the CSV into a pandas DataFrame
            df = pd.read_csv(io.BytesIO(csv_content), delimiter=',')

            # Check if 'api_key' column exists, if not, add it
            if 'api_key' not in df.columns:
                df.insert(0, 'api_key', '')
                print(f"Added 'api_key' column to {key}")

            # Convert the updated DataFrame back to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)

            # Upload the updated CSV back to S3
            s3.put_object(Bucket=bucket_name, Key=key, Body=csv_buffer.getvalue())
            print(f"Updated {key} in S3")

    print("Processing complete!")
