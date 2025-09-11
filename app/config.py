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
def get_settings() -> AnnoWorkerSettings:
    return AnnoWorkerSettings()
