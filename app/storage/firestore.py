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
    Save raw posts (with sentiment) to Firestore for archival.
    Each post gets its own document in 'post_archive' collection.

    Args:
        posts (list): List of full post dicts.
        timestamp (str): When this batch was fetched. If None, use current UTC.
    """
    collection = db.collection("post_archive")
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    batch = db.batch()
    for post in posts:
        doc_ref = collection.document(post["id"])
        post["archived_at"] = ts
        post["updatedAt"] = firestore.SERVER_TIMESTAMP
        batch.set(doc_ref, post)

    batch.commit()
    print(f"✅ Archived {len(posts)} posts to Firestore (post_archive/)")


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
