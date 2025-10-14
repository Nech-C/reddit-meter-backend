# file: app/jobs/runner.py
"""Entrypoint for orchestrating the Reddit sentiment ingestion pipeline."""

import argparse
import os
import logging
from datetime import datetime

from app.reddit.fetch import fetch_all_subreddit_posts_by_dict
from app.ml.inference import run_batch_inference
from app.processing.aggregate import compute_sentiment_average
from app.storage.firestore import default_repo
from app.storage.bigquery import default_bq_repo

from app.storage.bucket import default_bucket_repo
from app.ml.preprocessing import prepare_for_input
from app.logging_setup import setup_logging
from app.models.post import Post, Sentiment
from app import constants

setup_logging()
log = logging.getLogger("jobs.runner")


def main(
    method="hot",
    num_posts=15,
    num_comments=5,
    buffer=100,
    archive=True,
    snapshot=True,
    history=True,
):
    log.info(
        f"🚀 Starting sentiment pipeline (method={method}, posts={num_posts}, comments={num_comments})"
    )

    # Step 1: Fetch
    raw_data = fetch_all_subreddit_posts_by_dict(
        method=method,
        posts_per_subreddit=num_posts,
        comment_per_post=num_comments,
        fetch_buffer=buffer,
    )

    # Step 2: Flatten and prepare for inference
    all_posts: list[Post] = [
        post for cat in raw_data.values() for sub in cat for post in sub["posts"]
    ]

    texts = [
        prepare_for_input(
            post.post_title,
            post.post_text,
            [c.body for c in post.post_comments],
        )
        for post in all_posts
    ]

    log.info(f"🧠 Running inference on {len(texts)} posts...")
    predictions = run_batch_inference(texts)
    processing_timestamp = datetime.now(constants.TIMEZONE)
    for i, post in enumerate(all_posts):
        post.sentiment = Sentiment.model_validate(predictions[i])
        post.processing_timestamp = processing_timestamp
        post.sentiment_analysis_model = constants.DEFAULT_SENTIMENT_SOURCE
        if not post.post_subreddit:
            post.post_subreddit = "unknown"

    # Step 3: Aggregate
    aggregated = compute_sentiment_average(all_posts)

    # Step 4: Store
    repo = default_repo()
    bq_repo = default_bq_repo()
    if snapshot and os.environ.get("APP_ENV") != "test":
        repo.save_sentiment_summary(aggregated)
    if history:
        repo.save_sentiment_history(aggregated)
        bq_repo.insert_global_sentiment_history(aggregated)
    if archive:
        timestamp = processing_timestamp.isoformat()
        serialized_posts = [post.to_json_dict() for post in all_posts]
        default_bucket_repo().upload_json(serialized_posts, timestamp)

    log.info("✅ All steps completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Reddit sentiment collection and processing."
    )

    parser.add_argument(
        "--method",
        type=str,
        default="hot",
        choices=["hot", "new", "top"],
        help="Reddit listing method to use",
    )
    parser.add_argument("--posts", type=int, default=15, help="Posts per subreddit")
    parser.add_argument(
        "--comments", type=int, default=5, help="Minimum valid comments per post"
    )
    parser.add_argument(
        "--buffer", type=int, default=100, help="How many posts to sample per subreddit"
    )
    parser.add_argument(
        "--no-archive", action="store_true", help="Skip saving archived posts"
    )
    parser.add_argument(
        "--no-snapshot", action="store_true", help="Skip saving sentiment_current"
    )
    parser.add_argument(
        "--no-history", action="store_true", help="Skip saving sentiment_history"
    )

    args = parser.parse_args()

    main(
        method=args.method,
        num_posts=args.posts,
        num_comments=args.comments,
        buffer=args.buffer,
        archive=not args.no_archive,
        snapshot=not args.no_snapshot,
        history=not args.no_history,
    )
