# app/config.py
import os
from functools import lru_cache
from pathlib import Path


from google.api_core.retry import Retry
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

env_name = os.getenv("APP_ENV", "dev")
env_file = f".env.{env_name}"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_file, case_sensitive=True, extra="ignore"
    )
    GOOGLE_APPLICATION_CREDENTIALS: str | None = None
    API_OUTPUT_SCHEMA: str = "legacy"  # "legacy" or "new"


@lru_cache()
def get_app_settings() -> AppSettings:
    return AppSettings()


class AnnoWorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_file,
        case_sensitive=True,
        env_prefix="",
        env_nested_delimiter="__",
        secrets_dir=None,
        extra="ignore",
    )

    RUN_ID: str
    WORKER_ID: str = Field(default_factory=lambda: "worker")
    LEASE_MIN: int = 30
    LOAD_8BT: bool = True
    MAX_PROMPT_LEN: int = 1024
    MAX_NEW_TOKENS: int = 64
    CHUNK_SIZE: int = 1024
    BATCH_SIZE: int = 8

    SOURCE_HF_REPO: str = "Nech-C/reddit-sentiment"
    HF_TOKEN: str
    ANN_MODEL_ID: str = "Qwen/Qwen3-4B-Instruct-2507"

    GCS_BUCKET: str
    GCS_PREFIX: str = "annotations"
    FIRESTORE_DATABASE_ID: str = "sentiment-db"
    FIRESTORE_ANNO_COLLECTIONS: str = "annotation_runs"
    FIRESTORE_TASKS_SUBCOLLECTIONS: str = "tasks"

    @field_validator("RUN_ID", "GCS_BUCKET", "HF_TOKEN")
    def _require_non_empty(cls, v):
        if not v:
            raise ValueError("must be set and non-empty")
        return v


@lru_cache(maxsize=1)
def get_annotation_worker_settings() -> AnnoWorkerSettings:
    return AnnoWorkerSettings()


class StorageSettings(BaseSettings):
    """Storage settings for Firestore and GCS.

    Args:
        BaseSettings (BaseSettings): Base class for settings management.

    Raises:
        ValueError: if any required field is empty.

    Returns:
        StorageSettings: An instance of StorageSettings with loaded configuration.
    """

    model_config = SettingsConfigDict(
        env_file=env_file,
        case_sensitive=True,
        env_prefix="FIRESTORE_",
        env_nested_delimiter="__",
        secrets_dir=None,
        extra="ignore",
    )
    POST_ARCHIVE_COLLECTION_NAME: str
    SENTIMENT_HISTORY_COLLECTION_NAME: str
    CURRENT_SENTIMENT_COLLECTION_NAME: str
    HISTORY_RETRIEVAL_LIMIT: int = (24 / 4) * 30  # 30 days, history taken every 4 hrs
    DATABASE_ID: str
    GOOGLE_BUCKET_NAME: str

    @field_validator(
        "POST_ARCHIVE_COLLECTION_NAME",
        "SENTIMENT_HISTORY_COLLECTION_NAME",
        "CURRENT_SENTIMENT_COLLECTION_NAME",
        "DATABASE_ID",
        "GOOGLE_BUCKET_NAME",
    )
    def _require_non_empty(cls, v):
        if not v:
            raise ValueError("must be set and non-empty")
        return v


@lru_cache(maxsize=1)
def get_storage_settings() -> StorageSettings:
    """Get storage settings with caching.

    Returns:
        StorageSettings: An instance of StorageSettings with loaded configuration.
    """
    return StorageSettings()


class InferenceSettings(BaseSettings):
    """Settings for model inference.

    Args:
        BaseSettings (BaseSettings): Base class for settings management.
    """

    model_config = SettingsConfigDict(
        env_file=env_file,
        case_sensitive=True,
        env_prefix="",
        env_nested_delimiter="__",
        secrets_dir=None,
        extra="ignore",
    )

    BATCH_MAX_TOKENS: int = 512
    SENTIMENT_MODEL_ID: str = "bhadresh-savani/distilbert-base-uncased-emotion"


@lru_cache(maxsize=1)
def get_inference_settings() -> InferenceSettings:
    """Get inference settings with caching.

    Returns:
        InferenceSettings: An instance of InferenceSettings with loaded configuration.
    """
    return InferenceSettings()


class RedditSettings(BaseSettings):
    """Settings required for interacting with the Reddit API."""

    model_config = SettingsConfigDict(
        env_file=env_file,
        case_sensitive=True,
        env_prefix="REDDIT_",
        env_nested_delimiter="__",
        secrets_dir=None,
        extra="ignore",
    )

    CLIENT_ID: str
    CLIENT_SECRET: str
    PASSWORD: str
    USER_AGENT: str
    USERNAME: str
    RATELIMIT_SECONDS: int = 600
    SUBREDDIT_JSON_PATH: str

    @field_validator(
        "CLIENT_ID",
        "CLIENT_SECRET",
        "PASSWORD",
        "USER_AGENT",
        "USERNAME",
        "SUBREDDIT_JSON_PATH",
    )
    def _require_non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("must be set and non-empty")
        return value

    @field_validator("RATELIMIT_SECONDS")
    def _validate_ratelimit(cls, value: int) -> int:
        if value < 0:
            raise ValueError("ratelimit must be non-negative")
        return value

    @field_validator("SUBREDDIT_JSON_PATH")
    def _validate_path_exists(cls, value: str) -> str:
        path = Path(value)
        if not path.exists():
            raise ValueError(f"subreddit JSON file not found: {value}")
        return value


@lru_cache(maxsize=1)
def get_reddit_settings() -> RedditSettings:
    """Return cached Reddit API settings."""

    return RedditSettings()


class BigQuerySettings(BaseSettings):
    """Settings for BigQuery"""

    model_config = SettingsConfigDict(
        env_file=env_file,
        case_sensitive=True,
        env_prefix="",
        extra="ignore",
    )
    bq_dataset: str = Field(alias="BIGQUERY_DATASET_ID")
    bq_global_sentiment_history_table: str = Field(
        alias="BIGQUERY_GLOBAL_SENTIMENT_HISTORY_TABLE"
    )
    retry: Retry = Retry(
        initial=1.0,
        maximum=30.0,
        multiplier=2.0,
        deadline=60.0,
    )


@lru_cache(maxsize=1)
def get_bigquery_settings() -> BigQuerySettings:
    """return cached BigQuery settings"""
    return BigQuerySettings()
