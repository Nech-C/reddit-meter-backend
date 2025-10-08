import json
import logging
from typing import Union
from functools import lru_cache

from google.cloud import storage

from app.logging_setup import setup_logging
from app.config import StorageSettings, get_storage_settings

setup_logging()
log = logging.getLogger("storage.bucket")


class BucketRepo:
    def __init__(self, settings: StorageSettings = None, client: storage.Client = None):
        self.s = settings if settings else get_storage_settings()
        self.client = client if client else storage.Client()

    def upload_json(
        self, json_data: Union[list, dict], blob_name: str, bucket_name: str = None
    ):
        """Upload a JSON file to GCS

        Args:
            json_data (dict|list): Json object to upload
            blob_name (str): path in the bucket
            bucket_name (str): GCS bucket name
        """
        if bucket_name is None:
            bucket_name = self.s.GOOGLE_BUCKET_NAME
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        json_str = json.dumps(json_data, indent=2)
        blob.upload_from_string(json_str, content_type="application/json")

        print(f"✅ Uploaded JSON to gs://{bucket_name}/{blob_name}")


@lru_cache(maxsize=1)
def default_bucket_repo() -> BucketRepo:
    return BucketRepo()
