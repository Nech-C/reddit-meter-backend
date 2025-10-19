# File: app/storage/firestore.py
"""Firestore repository abstractions for storing Reddit sentiment data."""

import logging
import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Sequence, List

from google.api_core.retry import Retry
from google.cloud import firestore
from pydantic import ValidationError

from app.config import StorageSettings, get_storage_settings, get_app_settings
from app.logging_setup import setup_logging
from app.models.post import Post, TopSentimentContributor, SentimentSummary
from app import constants

setup_logging()
log = logging.getLogger("storage.firestore")


app_settings = get_app_settings()
if app_settings.GOOGLE_APPLICATION_CREDENTIALS:
    # only set if not already set (idempotent)
    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS", app_settings.GOOGLE_APPLICATION_CREDENTIALS
    )


## -------------------------------- ##
## Temporary legacy schema handling ##
## -------------------------------- ##


def _to_new_summary(raw: dict) -> dict:
    """Accept old/new doc, return NEW schema dict."""
    if not isinstance(raw, dict):
        return {"error": "Invalid document"}

    try:
        if SentimentSummary.model_validate(raw):
            return raw
    except Exception:
        log.exception(
            "_to_new_summary receives a json dict cannot be validated using SentimentSummary"
            "attemping conversion"
        )

    # derive top_contributors if old key exists
    tc = raw.get("top_contributors")
    if tc is None and isinstance(raw.get("_top_contributor"), dict):
        tc_list: List[TopSentimentContributor] = []
        for emotion, posts in raw["_top_contributor"].items():
            post_models = [Post.model_validate(p) for p in posts]
            tc_list.append(
                TopSentimentContributor(emotion=emotion, top_posts=post_models)
            )
        tc = tc_list
    elif tc is None:
        tc = []

    base = {
        k: raw.get(k) for k in ("joy", "sadness", "anger", "fear", "love", "surprise")
    }
    base["top_contributors"] = tc
    base["timestamp"] = raw["timestamp"]
    base["updatedAt"] = raw["updatedAt"]
    summary = SentimentSummary.model_validate(base)
    out = summary.model_dump(mode="python", exclude_none=False)

    return out


def _post_to_legacy_dict(p: dict) -> dict:
    """Map NEW post dict to legacy field names expected by old FE."""
    # p is already JSON-like (from model_dump(mode="json"))
    return {
        "id": p.get("post_id"),
        "url": p.get("post_url"),
        "title": p.get("post_title"),
        "text": p.get("post_text") or "",
        "created": p.get("post_created_ts"),
        "num_comments": p.get("post_comment_count"),
        "score": p.get("post_score"),
        "subreddit": p.get("post_subreddit"),
        "comments": p.get("post_comments") or [],
        "contribution": p.get("contribution"),
        "sentiment": p.get("sentiment"),
        "processing_timestamp": p.get("processing_timestamp"),
        "sentiment_source_model": p.get("sentiment_analysis_model", None),
        "sentiment_model_version": p.get("sentiment_model_version", None),
    }


def _to_legacy_summary(new_summary: dict) -> dict:
    """Take NEW schema dict and return the legacy shape expected by the FE."""
    legacy = {
        "joy": new_summary.get("joy", 0.0),
        "sadness": new_summary.get("sadness", 0.0),
        "anger": new_summary.get("anger", 0.0),
        "fear": new_summary.get("fear", 0.0),
        "love": new_summary.get("love", 0.0),
        "surprise": new_summary.get("surprise", 0.0),
    }

    # Build _top_contributor: {emotion: [posts...]}
    tcs = new_summary.get("top_contributors") or []
    top_map: Dict[str, List[dict]] = {}
    for tc in tcs:
        emotion = tc.get("emotion")
        posts = tc.get("top_posts") or []
        if not emotion:
            continue
        top_map.setdefault(emotion, []).extend(_post_to_legacy_dict(p) for p in posts)

    legacy["_top_contributor"] = top_map
    legacy["timestamp"] = new_summary["timestamp"]
    legacy["updatedAt"] = new_summary["updatedAt"]
    return legacy


## ------------------------------------- ##
## Temporary legacy schema handling(end) ##
## ------------------------------------- ##


class FirestoreRepo:
    def __init__(
        self,
        settings: StorageSettings | None = None,
        db: firestore.Client | None = None,
    ):
        self.s = settings if settings else get_storage_settings()
        self.db = db if db else firestore.Client(database=self.s.DATABASE_ID)
        self._retry = Retry(deadline=30.0)

    def save_sentiment_summary(self, aggregated_sentiment: SentimentSummary) -> None:
        """
        Save current snapshot of Reddit sentiment to Firestore (sentiment_current/global).
        """
        now = datetime.now(constants.TIMEZONE)
        try:
            SentimentSummary.model_validate(aggregated_sentiment)
        except ValidationError:
            log.error(
                "save_sentiment_summary receives aggregated_sentiment that fails to validate"
            )
        try:
            doc_ref = self.db.collection(
                self.s.CURRENT_SENTIMENT_COLLECTION_NAME
            ).document("global")

            payload = {
                **aggregated_sentiment.model_dump(mode="python", exclude_none=True),
                "timestamp": now,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            }
            doc_ref.set(payload, retry=self._retry)
            log.info("✅ Saved sentiment snapshot to Firestore (current/global).")
        except Exception:
            log.exception("Failed to save sentiment snapshot")

    def _prepare_posts_for_storage(
        self, posts: Sequence[Post | dict], *, json_mode: bool
    ) -> list[dict]:
        """Normalize posts into dictionaries for Firestore or JSON archives."""

        serialized = []
        for post in posts:
            if isinstance(post, Post):
                dump = post.to_json_dict() if json_mode else post.to_python_dict()
            else:
                dump = post
            serialized.append(dump)
        return serialized

    def save_post_archive(
        self, posts: Sequence[Post | dict], timestamp: str = None
    ) -> None:
        """
        Save all posts from one job into a single Firestore document.
        Document name will be based on UTC timestamp: YYYYMMDDHH

        Args:
            posts (Sequence[Post | dict]): Posts (models or dictionaries) with
                associated sentiment scores.
            timestamp (str): Optional ISO 8601 timestamp string.
        """

        dt = (
            datetime.fromisoformat(timestamp)
            if timestamp
            else datetime.now(constants.TIMEZONE)
        )
        normalized_posts = self._prepare_posts_for_storage(posts, json_mode=False)
        hour_id = dt.strftime("%Y%m%d%H")  # e.g., 2025062713
        try:
            doc_ref = self.db.collection(self.s.POST_ARCHIVE_COLLECTION_NAME).document(
                hour_id
            )
            doc_ref.set(
                {
                    "posts": normalized_posts,
                    "count": len(normalized_posts),
                    "archived_at": dt.isoformat(),
                },
                retry=self._retry,
            )
            log.info(
                f"✅ Archived {len(normalized_posts)} posts to Firestore (post_archive/{hour_id})"
            )
        except Exception:
            log.exception("Failed to archive posts")

    def save_sentiment_history(self, aggregated_sentiment: SentimentSummary) -> None:
        """
        Save sentiment snapshot to a timestamped document in Firestore (sentiment_history/<hour>).
        Useful for tracking trends over time.
        """
        now = datetime.now(constants.TIMEZONE)
        hour_key = now.strftime("%Y-%m-%dT%H")
        try:
            SentimentSummary.model_validate(aggregated_sentiment)
        except ValidationError:
            log.error(
                "save_sentiment_history receives aggregated_sentiment that fails to validate"
            )
        payload = {
            **aggregated_sentiment.model_dump(mode="python", exclude_none=True),
            "timestamp": now,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
        try:
            doc_ref = self.db.collection(
                self.s.SENTIMENT_HISTORY_COLLECTION_NAME
            ).document(hour_key)
            doc_ref.set(payload, retry=self._retry)
            log.info(
                f"✅ Saved sentiment history snapshot to Firestore (sentiment_history/{hour_key})"
            )
        except Exception:
            log.exception("Failed to save sentiment history")

    def get_latest_sentiment(self) -> Dict:
        """
        Retrieve the latest snapshot from Firestore (sentiment_current/global).
        """
        try:
            doc = (
                self.db.collection(self.s.CURRENT_SENTIMENT_COLLECTION_NAME)
                .document("global")
                .get(retry=self._retry)
            )
            if not doc.exists:
                return {"error": "No sentiment data found."}
            # first normalize to NEW, then optionally downgrade to LEGACY
        except Exception:
            log.exception("Failed to read latest sentiment")
            return {"error": "Firestore read failed."}

        try:
            new_shape = _to_new_summary(doc.to_dict())
            return (
                _to_legacy_summary(new_shape)
                if app_settings.API_OUTPUT_SCHEMA == "legacy"
                else new_shape
            )
        except Exception:
            log.exception(
                f"failed to convert into valid format. Mode: {app_settings.API_OUTPUT_SCHEMA}"
            )

    def get_recent_sentiment_history(self, num_days: int) -> list[Dict]:
        """Retrieve the sentiment history from Firestore (sentiment_history).

        Args:
            num_records (int): The number of data points to retrieve.
        """
        now = datetime.now(constants.TIMEZONE)
        start_date = now - timedelta(days=num_days)
        try:
            docs = (
                self.db.collection(self.s.SENTIMENT_HISTORY_COLLECTION_NAME)
                .where("timestamp", ">=", start_date)
                .stream(retry=self._retry)
            )
            out = []
            for d in docs:
                new_shape = _to_new_summary(d.to_dict())
                out.append(
                    _to_legacy_summary(new_shape)
                    if app_settings.API_OUTPUT_SCHEMA == "legacy"
                    else new_shape
                )
            return out
        except Exception:
            log.exception("Failed to read sentiment history")
            return []

    def healthcheck(self):
        """Perform a simple healthcheck for App Engine warm-up call"""
        self.db.collection(self.s.CURRENT_SENTIMENT_COLLECTION_NAME).limit(1).stream()


@lru_cache(maxsize=1)
def default_repo() -> FirestoreRepo:
    return FirestoreRepo()
