# File: app/storage/firestore.py
"""Firestore repository abstractions for storing Reddit sentiment data."""

import logging
import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Sequence

from google.api_core.retry import Retry
from google.cloud import firestore

from app.config import StorageSettings, get_storage_settings, get_app_settings
from app.logging_setup import setup_logging
from app.models.post import Post
from app import constants

setup_logging()
log = logging.getLogger("storage.firestore")


app_settings = get_app_settings()
if app_settings.GOOGLE_APPLICATION_CREDENTIALS:
    # only set if not already set (idempotent)
    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS", app_settings.GOOGLE_APPLICATION_CREDENTIALS
    )


class FirestoreRepo:
    def __init__(
        self,
        settings: StorageSettings | None = None,
        db: firestore.Client | None = None,
    ):
        self.s = settings if settings else get_storage_settings()
        self.db = db if db else firestore.Client(database=self.s.DATABASE_ID)
        self._retry = Retry(deadline=30.0)

    def save_sentiment_summary(self, aggregated_sentiment: dict) -> None:
        """
        Save current snapshot of Reddit sentiment to Firestore (sentiment_current/global).
        """
        now = datetime.now(constants.TIMEZONE)
        try:
            doc_ref = self.db.collection(
                self.s.CURRENT_SENTIMENT_COLLECTION_NAME
            ).document("global")

            payload = {
                **aggregated_sentiment,
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

    def save_sentiment_history(self, aggregated_sentiment: dict) -> None:
        """
        Save sentiment snapshot to a timestamped document in Firestore (sentiment_history/<hour>).
        Useful for tracking trends over time.
        """
        now = datetime.now(constants.TIMEZONE)
        hour_key = now.strftime("%Y-%m-%dT%H")
        payload = {
            **aggregated_sentiment,
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
            if doc.exists:
                return doc.to_dict()
            return {"error": "No sentiment data found."}
        except Exception:
            log.exception("Failed to read latest sentiment")
            return {"error": "Firestore read failed."}

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

            return [doc.to_dict() for doc in docs]
        except Exception:
            log.exception("Failed to read sentiment history")
            return []

    def healthcheck(self):
        """Perform a simple healthcheck for App Engine warm-up call"""
        self.db.collection(self.s.CURRENT_SENTIMENT_COLLECTION_NAME).limit(1).stream()


@lru_cache(maxsize=1)
def default_repo() -> FirestoreRepo:
    return FirestoreRepo()
