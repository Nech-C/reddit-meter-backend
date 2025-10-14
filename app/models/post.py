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
    post_text_preview: Optional[str] = None
    # Prefer aware datetimes for easier downstream processing.
    post_created_ts: Optional[datetime] = Field(default=None, alias="created")
    # TODO: migrate to post_created_ts
    # make sure both frontend and backend compatibility
    # make sure the right format is used for all storage: bigquery, firestore, and gcs
    post_score: Optional[int] = Field(default=None, alias="score")

    post_comment_count: OptNonNegativeInt = Field(default=None, alias="num_comments")
    post_comments: List[PostComment] = Field(default_factory=list, alias="comments")

    post_subreddit: Optional[str] = Field(default=None, alias="subreddit")

    # Generated / processing metadata.
    contribution: float = None
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

        return self.model_dump(mode="json", exclude_none=True)

    def to_python_dict(self) -> dict:
        """Return a Python-native dictionary suitable for Firestore writes."""

        return self.model_dump(mode="python", exclude_none=True)

    def to_bq_dict(self, preview_length_limit=constants.DEFAULT_BQ_TEXT_PREVIEW_MAX):
        """
        Return a BigQuery-ready dict version of this Post.

        - Keeps all fields except post_text.
        - Adds post_text_preview (truncated).
        - Omits None fields for cleaner insert payloads.
        """
        dump = self.model_dump(mode="json", exclude_none=True)

        full_text = dump.get("post_text")
        if full_text:
            dump["post_text_preview"] = full_text[:preview_length_limit]
        else:
            dump["post_text_preview"] = None

        # Remove full text to avoid uploading big blobs to BigQuery
        dump.pop("post_text", None)

        return dump


class TopSentimentContributor(BaseModel):
    """Validated top contributors for a single sentiment"""

    model_config = ConfigDict(extra="ignore")
    emotion: str
    top_posts: List[Post]

    def to_bq_dict(self):
        dump = self.model_dump(mode="json", exclude_none=True, exclude={"top_posts"})
        dump["top_posts"] = list(map(Post.to_bq_dict, self.top_posts))
        return dump


class SentimentSummary(BaseModel):
    """Validated sentiment summary for Firestore and BigQuery"""

    model_config = ConfigDict(extra="ignore")

    joy: Probability = 0.0
    sadness: Probability = 0.0
    anger: Probability = 0.0
    fear: Probability = 0.0
    love: Probability = 0.0
    surprise: Probability = 0.0

    top_contributors: List[TopSentimentContributor]

    def to_bq_dict(self):
        dump = self.model_dump(
            mode="json", exclude_none=True, exclude={"top_contributors"}
        )
        dump["top_contributors"] = list(
            map(TopSentimentContributor.to_bq_dict, self.top_contributors)
        )
        return dump
