# File: app/storage/firestore.py
from datetime import datetime, timezone, timedelta
import os

from dotenv import load_dotenv
from google.cloud import firestore

env_name = os.getenv("APP_ENV", "test")
load_dotenv(f".env.{env_name}")
HISTORY_RETRIEVAL_LIMIT = (24 / 4) * 30  # 30 days, history taken every 4 hrs
POST_ARCHIVE_COLLECTION_NAME = os.getenv("FIRESTORE_POST_ARCHIVE_COLLECTION_NAME")
SENTIMENT_HISTORY_COLLECTION_NAME = os.getenv(
    "FIRESTORE_SENTIMENT_HISTORY_COLLECTION_NAME"
)
CURRENT_SENTIMENT_COLLECTION_NAME = os.getenv(
    "FIRESTORE_CURRENT_SENTIMENT_COLLECTION_NAME"
)

db = firestore.Client(database=os.getenv("FIRESTORE_DATABASE_ID"))


def save_sentiment_summary(aggregated_sentiment: dict):
    """
    Save current snapshot of Reddit sentiment to Firestore (sentiment_current/global).
    """
    doc_ref = db.collection(CURRENT_SENTIMENT_COLLECTION_NAME).document("global")
    aggregated_sentiment["timestamp"] = datetime.now(timezone.utc).isoformat()
    aggregated_sentiment["updatedAt"] = firestore.SERVER_TIMESTAMP
    doc_ref.set(aggregated_sentiment)
    print("✅ Saved sentiment snapshot to Firestore.")


def save_post_archive(posts: list[dict], timestamp: str = None):
    """
    Save all posts from one job into a single Firestore document.
    Document name will be based on UTC timestamp: YYYYMMDDHH

    Args:
        posts (list): List of post dicts (with sentiment).
        timestamp (str): Optional ISO 8601 timestamp string.
    """
    dt = datetime.fromisoformat(timestamp) if timestamp else datetime.now(timezone.utc)
    doc_id = dt.strftime("%Y%m%d%H")  # e.g., 2025062713
    doc_ref = db.collection(POST_ARCHIVE_COLLECTION_NAME).document(doc_id)

    doc_ref.set(
        {
            "posts": posts,
            "count": len(posts),
            "archieved_at": dt.isoformat(),
        }
    )
    print(f"✅ Archived {len(posts)} posts to Firestore (post_archive/{doc_id})")


def save_sentiment_history(aggregated_sentiment: dict):
    """
    Save sentiment snapshot to a timestamped document in Firestore (sentiment_history/<hour>).
    Useful for tracking trends over time.
    """
    hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")  # e.g., 2025-06-25T15
    doc_ref = db.collection(SENTIMENT_HISTORY_COLLECTION_NAME).document(hour_key)

    aggregated_sentiment["timestamp"] = datetime.now(timezone.utc).isoformat()
    aggregated_sentiment["updatedAt"] = firestore.SERVER_TIMESTAMP

    doc_ref.set(aggregated_sentiment)
    print(
        f"✅ Saved sentiment history snapshot to Firestore (sentiment_history/{hour_key})"
    )


def get_latest_sentiment():
    """
    Retrieve the latest snapshot from Firestore (sentiment_current/global).
    """
    doc = db.collection(CURRENT_SENTIMENT_COLLECTION_NAME).document("global").get()
    if doc.exists:
        return doc.to_dict()
    return {"error": "No sentiment data found."}


def get_recent_sentiment_history(num_days) -> list:
    """Retrieve the sentiment history from Firestore (sentiment_history).

    Args:
        num_records (int): The number of data points to retrieve.
    """
    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=num_days)).isoformat()

    docs = (
        db.collection(SENTIMENT_HISTORY_COLLECTION_NAME)
        .where("timestamp", ">=", start_date)
        .stream()
    )

    return [doc.to_dict() for doc in docs]
