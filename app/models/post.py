# app/models/post.py
"""Typed representations of Reddit posts, comments and sentiment scores."""

from typing import Optional, List, Annotated, Any
from datetime import datetime
import math

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)

import app.constants as constants

NonNegativeInt = Annotated[int, Field(ge=0)]
OptNonNegativeInt = Annotated[Optional[int], Field(ge=0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class PostComment(BaseModel):
    """Minimal, validated representation of a Reddit comment."""

    body: str
    author: Optional[str] = None
    # Reddit returns non-negative scores; enforce to guard against invalid data.
    score: OptNonNegativeInt = None
    created_utc: Optional[datetime] = None  # unix seconds

    model_config = ConfigDict(extra="ignore")


class Sentiment(BaseModel):
    joy: Probability = 0.0
    sadness: Probability = 0.0
    anger: Probability = 0.0
    fear: Probability = 0.0
    love: Probability = 0.0
    surprise: Probability = 0.0

    model_config = ConfigDict(extra="ignore")

    # run after field-level validation on the instance
    @model_validator(mode="after")
    def _normalize_if_needed(self) -> "Sentiment":
        vals = [self.joy, self.sadness, self.anger, self.fear, self.love, self.surprise]
        total = sum(vals)
        # nothing predicted
        if total == 0 or math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-9):
            return self
        # normalize in-place and return instance
        self.joy /= total
        self.sadness /= total
        self.anger /= total
        self.fear /= total
        self.love /= total
        self.surprise /= total
        return self


class Post(BaseModel):
    """Validated Reddit post including metadata, comments, and sentiment."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # Source fields set by the Reddit API fetcher.
    post_id: Optional[str] = Field(default=None, alias="id")
    post_url: Optional[str] = Field(default=None, alias="url")
    post_title: Optional[str] = Field(default=None, alias="title")
    post_text: Optional[str] = Field(default=None, alias="text")
    # Prefer aware datetimes for easier downstream processing.
    post_created_ts: Optional[datetime] = Field(default=None, alias="created")
    # TODO: migrate to post_created_ts
    # make sure both frontend and backend compatibility
    # make sure the right format is used for all storage: bigquery, firestore, and gcs
    score: Optional[int] = None

    post_comment_count: OptNonNegativeInt = Field(default=None, alias="num_comments")
    post_comments: List[PostComment] = Field(default_factory=list, alias="comments")

    post_subreddit: Optional[str] = Field(default=None, alias="subreddit")

    # Generated / processing metadata.
    contribution: OptNonNegativeInt = None
    sentiment: Optional[Sentiment] = None
    processing_timestamp: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(constants.TIMEZONE)
    )
    sentiment_analysis_model: Optional[str] = Field(
        default=None, alias="sentiment_source_model"
    )
    sentiment_model_version: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_keys(cls, data: Any):
        """Support legacy dictionary keys from the previous implementation."""

        if isinstance(data, dict):
            data = data.copy()
            if "created_utc" in data and "created" not in data:
                data["created"] = data["created_utc"]
        return data

    @field_validator("post_created_ts", mode="before")
    @classmethod
    def _coerce_created_ts(cls, v: Any):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=constants.TIMEZONE)
        return v

    @field_validator("processing_timestamp", mode="before")
    @classmethod
    def _coerce_processing_ts(cls, v: Any):
        # If caller omitted the field, Pydantic will call the default_factory and v will be a datetime,
        # or in some cases the validator may not be called — but we only need to handle numeric inputs here.
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=constants.TIMEZONE)
        return v

    @model_validator(mode="after")
    def _check_timestamps(self):
        # only run if both present
        if self.processing_timestamp and self.post_created_ts:
            if self.processing_timestamp < self.post_created_ts:
                raise ValueError(
                    "processing_timestamp cannot be earlier than post_created_ts"
                )
        return self

    def to_json_dict(self) -> dict:
        """Return a JSON-serialisable dictionary representation of the post."""

        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def to_python_dict(self) -> dict:
        """Return a Python-native dictionary suitable for Firestore writes."""

        return self.model_dump(mode="python", by_alias=True, exclude_none=True)


class TopContributor(BaseModel):
    """Validated top contributing Reddit post including metadata, comments, sentiment, and contribution."""

    # TODO: this class is a temp solution. fix this later!!!
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # Source fields set by the Reddit API fetcher.
    post_id: Optional[str] = Field(default=None, alias="id")
    post_url: Optional[str] = Field(default=None, alias="url")
    post_title: Optional[str] = Field(default=None, alias="title")
    post_text: Optional[str] = Field(default=None, alias="text")
    # Prefer aware datetimes for easier downstream processing.
    post_created_ts: Optional[datetime] = Field(default=None, alias="created")
    # TODO: migrate to post_created_ts
    # make sure both frontend and backend compatibility
    # make sure the right format is used for all storage: bigquery, firestore, and gcs
    score: Optional[int] = None

    post_comment_count: OptNonNegativeInt = Field(default=None, alias="num_comments")
    post_comments: List[PostComment] = Field(default_factory=list, alias="comments")

    post_subreddit: Optional[str] = Field(default=None, alias="subreddit")

    # Generated / processing metadata.
    contribution: OptNonNegativeInt = None
    sentiment: Optional[Sentiment] = None
    processing_timestamp: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(constants.TIMEZONE)
    )
    sentiment_analysis_model: Optional[str] = Field(
        default=None, alias="sentiment_source_model"
    )
    sentiment_model_version: Optional[str] = None

    contribution: float

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_keys(cls, data: Any):
        """Support legacy dictionary keys from the previous implementation."""

        if isinstance(data, dict):
            data = data.copy()
            if "created_utc" in data and "created" not in data:
                data["created"] = data["created_utc"]
        return data

    @field_validator("post_created_ts", mode="before")
    @classmethod
    def _coerce_created_ts(cls, v: Any):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=constants.TIMEZONE)
        return v

    @field_validator("processing_timestamp", mode="before")
    @classmethod
    def _coerce_processing_ts(cls, v: Any):
        # If caller omitted the field, Pydantic will call the default_factory and v will be a datetime,
        # or in some cases the validator may not be called — but we only need to handle numeric inputs here.
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=constants.TIMEZONE)
        return v

    @model_validator(mode="after")
    def _check_timestamps(self):
        # only run if both present
        if self.processing_timestamp and self.post_created_ts:
            if self.processing_timestamp < self.post_created_ts:
                raise ValueError(
                    "processing_timestamp cannot be earlier than post_created_ts"
                )
        return self

    def to_json_dict(self) -> dict:
        """Return a JSON-serialisable dictionary representation of the post."""

        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def to_python_dict(self) -> dict:
        """Return a Python-native dictionary suitable for Firestore writes."""

        return self.model_dump(mode="python", by_alias=True, exclude_none=True)


class TopSentimentContributors(BaseModel):
    """Validated top contributors for a single sentiment"""

    model_config = ConfigDict(extra="ignore")
    joy: Optional[List[TopContributor]] = None
    sadness: Optional[List[TopContributor]] = None
    anger: Optional[List[TopContributor]] = None
    fear: Optional[List[TopContributor]] = None
    love: Optional[List[TopContributor]] = None
    surprise: Optional[List[TopContributor]] = None

    def model_dump(self, *args, **kwargs):
        kwargs["exclude_none"] = True
        return super().model_dump(*args, **kwargs)


class SentimentSummary(BaseModel):
    """Validated sentiment summary for Firestore and BigQuery"""

    model_config = ConfigDict(extra="ignore")

    joy: Probability = 0.0
    sadness: Probability = 0.0
    anger: Probability = 0.0
    fear: Probability = 0.0
    love: Probability = 0.0
    surprise: Probability = 0.0

    top_contributor: TopSentimentContributors
