import pytest

from app import config


@pytest.fixture(autouse=True)
def clear_caches():
    config.get_annotation_worker_settings.cache_clear()
    config.get_storage_settings.cache_clear()
    config.get_inference_settings.cache_clear()
    config.get_bigquery_settings.cache_clear()
    yield
    config.get_annotation_worker_settings.cache_clear()
    config.get_storage_settings.cache_clear()
    config.get_inference_settings.cache_clear()
    config.get_bigquery_settings.cache_clear()


def test_get_annotation_worker_settings_reads_env(monkeypatch):
    monkeypatch.setenv("RUN_ID", "run-123")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("GCS_BUCKET", "bucket")

    settings = config.get_annotation_worker_settings()
    assert settings.RUN_ID == "run-123"
    assert settings.HF_TOKEN == "secret"
    assert config.get_annotation_worker_settings() is settings

    # mutate env to confirm caching behaviour
    monkeypatch.setenv("RUN_ID", "other")
    assert config.get_annotation_worker_settings().RUN_ID == "run-123"


def test_annotation_worker_settings_validation():
    with pytest.raises(ValueError):
        config.AnnoWorkerSettings(RUN_ID="", HF_TOKEN="token", GCS_BUCKET="bucket")


@pytest.fixture
def storage_env(monkeypatch):
    monkeypatch.setenv("FIRESTORE_POST_ARCHIVE_COLLECTION_NAME", "posts")
    monkeypatch.setenv("FIRESTORE_SENTIMENT_HISTORY_COLLECTION_NAME", "history")
    monkeypatch.setenv("FIRESTORE_CURRENT_SENTIMENT_COLLECTION_NAME", "current")
    monkeypatch.setenv("FIRESTORE_FIRESTORE_DATABASE_ID", "db")
    monkeypatch.setenv("FIRESTORE_GOOGLE_BUCKET_NAME", "bucket")
    return monkeypatch


def test_get_storage_settings_reads_env(storage_env):
    settings = config.get_storage_settings()
    assert settings.POST_ARCHIVE_COLLECTION_NAME == "posts"
    assert settings.GOOGLE_BUCKET_NAME == "bucket"
    assert config.get_storage_settings() is settings

    storage_env.setenv("FIRESTORE_GOOGLE_BUCKET_NAME", "other")
    # still cached
    assert config.get_storage_settings().GOOGLE_BUCKET_NAME == "bucket"


def test_storage_settings_validation():
    with pytest.raises(ValueError):
        config.StorageSettings(
            POST_ARCHIVE_COLLECTION_NAME="",
            SENTIMENT_HISTORY_COLLECTION_NAME="history",
            CURRENT_SENTIMENT_COLLECTION_NAME="current",
            FIRESTORE_DATABASE_ID="db",
            GOOGLE_BUCKET_NAME="bucket",
        )


def test_get_inference_settings(monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODEL_ID", "custom-model")
    settings = config.get_inference_settings()
    assert settings.SENTIMENT_MODEL_ID == "custom-model"
    assert config.get_inference_settings() is settings

    monkeypatch.setenv("SENTIMENT_MODEL_ID", "other")
    assert config.get_inference_settings().SENTIMENT_MODEL_ID == "custom-model"


def test_bq_settings():
    """Ensure BigQuerySettings work properly"""
    # all properties are available
    settings = config.BigQuerySettings()

    assert settings.bq_dataset == "test_bq_dataset"
    assert settings.bq_global_sentiment_history_table == "test_global_sentiment_table"


# def test_bq_settings_failure(monkeypatch):
#     """Ensure all required fields exist"""
#     monkeypatch.delenv("BIGQUERY_DATASET_ID", raising=False)
#     monkeypatch.delenv("BIGQUERY_GLOBAL_SENTIMENT_HISTORY_TABLE", raising=False)

#     with pytest.raises(ValidationError):
#         settings = config.BigQuerySettings()
