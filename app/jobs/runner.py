# file: app/jobs/runner.py
import argparse
import os
import logging
from datetime import datetime, timezone

from app.reddit.fetch import fetch_all_subreddit_posts_by_dict
from app.ml.inference import run_batch_inference
from app.processing.aggregate import compute_sentiment_average
from app.storage.firestore import default_repo

from app.storage.bucket import default_bucket_repo
from app.ml.preprocessing import prepare_for_input
from app.logging_setup import setup_logging

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
    all_posts = [
        post for cat in raw_data.values() for sub in cat for post in sub["posts"]
    ]

    texts = [
        prepare_for_input(
            post["title"], post["text"], [c["body"] for c in post["comments"]]
        )
        for post in all_posts
    ]

    log.info(f"🧠 Running inference on {len(texts)} posts...")
    predictions = run_batch_inference(texts)
    for i, post in enumerate(all_posts):
        post["sentiment"] = predictions[i]
        post.update(
            {
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "sentiment_source_model": "bert",
                "subreddit": post.get("subreddit", "unknown"),
            }
        )

    # Step 3: Aggregate
    aggregated = compute_sentiment_average(all_posts)

    # Step 4: Store
    repo = default_repo()
    if snapshot and os.environ.get("APP_ENV") != "test":
        repo.save_sentiment_summary(aggregated)
    if history:
        repo.save_sentiment_history(aggregated)
    if archive:
        timestamp = datetime.now(timezone.utc).isoformat()
        default_bucket_repo().upload_json(all_posts, timestamp)

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
