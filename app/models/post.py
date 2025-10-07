# app/models/post.py
from typing import Optional, List, Annotated, Any
from datetime import datetime
import math

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    HttpUrl,
)

import app.constants as constants

NonNegativeInt = Annotated[int, Field(ge=0)]
OptNonNegativeInt = Annotated[Optional[int], Field(ge=0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class PostComment(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    body: str
    author: Optional[str] = None
    score: Annotated[Optional[int], Field(ge=-0)] = None
    created_utc: Optional[float] = None  # unix seconds

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
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # source fields (make them optional with defaults if callers may omit)
    post_id: Optional[str] = None
    post_url: Optional[HttpUrl] = None
    post_title: Optional[str] = None
    # prefer datetime for easier handling
    post_created_ts: Optional[datetime] = None

    score: OptNonNegativeInt = None

    post_comment_count: OptNonNegativeInt = None
    post_comments: List[PostComment] = Field(default_factory=list)

    post_subreddit: Optional[str] = None

    # generated / processing
    contribution: OptNonNegativeInt = None
    sentiment: Optional[Sentiment] = None
    processing_timestamp: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(constants.TIMEZONE)
    )
    sentiment_analysis_model: Optional[str] = None
    sentiment_model_version: Optional[str] = None

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
