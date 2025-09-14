# app/config.py
import os
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

env_name = os.getenv("APP_ENV", "dev")
env_file = f".env.{env_name}"


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
    model_config = SettingsConfigDict(
        env_file=env_file,
        case_sensitive=True,
        env_prefix="",
        env_nested_delimiter="__",
        secrets_dir=None,
        extra="ignore",
    )
    POST_ARCHIVE_COLLECTION_NAME: str
    SENTIMENT_HISTORY_COLLECTION_NAME: str
    CURRENT_SENTIMENT_COLLECTION_NAME: str
    HISTORY_RETRIEVAL_LIMIT: int = (24 / 4) * 30  # 30 days, history taken every 4 hrs
    FIRESTORE_DATABASE_ID: str

    @field_validator(
        "POST_ARCHIVE_COLLECTION_NAME",
        "SENTIMENT_HISTORY_COLLECTION_NAME",
        "CURRENT_SENTIMENT_COLLECTION_NAME",
        "FIRESTORE_DATABASE_ID",
    )
    def _require_non_empty(cls, v):
        if not v:
            raise ValueError("must be set and non-empty")
        return v
