import os
import json
from typing import Union
from google.cloud import storage
from dotenv import load_dotenv

env_name = os.getenv("APP_ENV", "dev")
load_dotenv(f".env.{env_name}")

client = storage.Client()


def upload_json(json_data: Union[list, dict], blob_name: str, bucket_name: str = None):
    """Upload a JSON file to GCS

    Args:
        json_data (dict|list): Json object to upload
        blob_name (str): path in the bucket
        bucket_name (str): GCS bucket name
    """
    if bucket_name is None:
        bucket_name = os.getenv("GOOGLE_BUCKET_NAME")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    print(json_data)
    json_str = json.dumps(json_data, indent=2)
    print(json_str)
    blob.upload_from_string(json_str, content_type="application/json")

    print(f"✅ Uploaded JSON to gs://{bucket_name}/{blob_name}")
