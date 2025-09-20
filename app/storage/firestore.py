# File: app/storage/firestore.py
import logging
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Dict

from google.api_core.retry import Retry
from google.cloud import firestore

from app.config import StorageSettings, get_storage_settings
from app.logging_setup import setup_logging

setup_logging()
log = logging.getLogger("storage.firestore")


class FirestoreRepo:
    def __init__(
        self,
        settings: StorageSettings | None = None,
        db: firestore.Client | None = None,
    ):
        self.s = settings if settings else get_storage_settings()
        self.db = db if db else firestore.Client(database=self.s.FIRESTORE_DATABASE_ID)
        self._retry = Retry(deadline=30.0)

    def save_sentiment_summary(self, aggregated_sentiment: dict) -> None:
        """
        Save current snapshot of Reddit sentiment to Firestore (sentiment_current/global).
        """
        now = datetime.now(timezone.utc)
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

    def save_post_archive(self, posts: list[dict], timestamp: str = None) -> None:
        """
        Save all posts from one job into a single Firestore document.
        Document name will be based on UTC timestamp: YYYYMMDDHH

        Args:
            posts (list): List of post dicts (with sentiment).
            timestamp (str): Optional ISO 8601 timestamp string.
        """

        dt = (
            datetime.fromisoformat(timestamp)
            if timestamp
            else datetime.now(timezone.utc)
        )
        hour_id = dt.strftime("%Y%m%d%H")  # e.g., 2025062713
        try:
            doc_ref = self.db.collection(self.s.POST_ARCHIVE_COLLECTION_NAME).document(
                hour_id
            )
            doc_ref.set(
                {
                    "posts": posts,
                    "count": len(posts),
                    "archived_at": dt.isoformat(),
                },
                retry=self._retry,
            )
            log.info(
                f"✅ Archived {len(posts)} posts to Firestore (post_archive/{hour_id})"
            )
        except Exception:
            log.exception("Failed to archive posts")

    def save_sentiment_history(self, aggregated_sentiment: dict) -> None:
        """
        Save sentiment snapshot to a timestamped document in Firestore (sentiment_history/<hour>).
        Useful for tracking trends over time.
        """
        now = datetime.now(timezone.utc)
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
        now = datetime.now(timezone.utc)
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
