# File: app/storage/firestore.py
from datetime import datetime, timezone
import os

from dotenv import load_dotenv
from google.cloud import firestore

env_name = os.getenv("APP_ENV", "dev")
load_dotenv(f".env.{env_name}")

db = firestore.Client(database=os.getenv("FIRESTORE_DATABASE_ID"))


def save_sentiment_summary(aggregated_sentiment: dict):
    """
    Save current snapshot of Reddit sentiment to Firestore (sentiment_current/global).
    """
    doc_ref = db.collection("sentiment_current").document("global")
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
    doc_ref = db.collection("post_archive").document(doc_id)

    for post in posts:
        post["archived_at"] = dt.isoformat()
        post["updatedAt"] = firestore.SERVER_TIMESTAMP

    doc_ref.set({"posts": posts, "count": len(posts), "archived_at": dt.isoformat(), "updatedAt": firestore.SERVER_TIMESTAMP})
    print(f"✅ Archived {len(posts)} posts to Firestore (post_archive/{doc_id})")



def save_sentiment_history(aggregated_sentiment: dict):
    """
    Save sentiment snapshot to a timestamped document in Firestore (sentiment_history/<hour>).
    Useful for tracking trends over time.
    """
    hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")  # e.g., 2025-06-25T15
    doc_ref = db.collection("sentiment_history").document(hour_key)

    aggregated_sentiment["timestamp"] = datetime.now(timezone.utc).isoformat()
    aggregated_sentiment["updatedAt"] = firestore.SERVER_TIMESTAMP

    doc_ref.set(aggregated_sentiment)
    print(
        f"✅ Saved sentiment history snapshot to Firestore (sentiment_history/{hour_key})"
    )
