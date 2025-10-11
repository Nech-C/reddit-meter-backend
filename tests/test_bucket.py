import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.storage import bucket


def test_bucket_repo_defaults_use_helpers(monkeypatch):
    sentinel_settings = object()
    sentinel_client = object()

    monkeypatch.setattr(bucket, "get_storage_settings", lambda: sentinel_settings)
    monkeypatch.setattr(bucket.storage, "Client", lambda: sentinel_client)

    repo = bucket.BucketRepo(settings=None, client=None)
    assert repo.s is sentinel_settings
    assert repo.client is sentinel_client


def test_upload_json_uses_env_bucket(monkeypatch):
    mock_client = MagicMock()
    mock_bucket = mock_client.bucket.return_value
    mock_blob = mock_bucket.blob.return_value

    repo = bucket.BucketRepo(
        settings=SimpleNamespace(GOOGLE_BUCKET_NAME="env-bucket"), client=mock_client
    )

    monkeypatch.setenv("GOOGLE_BUCKET_NAME", "env-bucket")

    repo.upload_json({"key": "value"}, "path/to/blob.json")

    mock_client.bucket.assert_called_once_with("env-bucket")
    mock_bucket.blob.assert_called_once_with("path/to/blob.json")

    (payload,), kwargs = mock_blob.upload_from_string.call_args
    assert json.loads(payload) == {"key": "value"}
    assert kwargs["content_type"] == "application/json"


def test_upload_json_with_explicit_bucket():
    mock_client = MagicMock()
    mock_bucket = mock_client.bucket.return_value
    mock_blob = mock_bucket.blob.return_value

    repo = bucket.BucketRepo(
        settings=SimpleNamespace(GOOGLE_BUCKET_NAME="bucket"), client=mock_client
    )

    repo.upload_json({"n": 1}, "blob.json", bucket_name="custom-bucket")

    mock_client.bucket.assert_called_once_with("custom-bucket")
    mock_blob.upload_from_string.assert_called_once()
