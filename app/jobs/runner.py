# file: app/jobs/runner.py
import argparse

from app.reddit.fetch import fetch_all_subreddit_posts_by_dict
from app.ml.inference import run_batch_inference
from app.processing.aggregate import compute_sentiment_average
from app.storage.firestore import (
    save_sentiment_summary,
    save_sentiment_history,
)


def main(
    method="hot",
    posts=15,
    comments=5,
    buffer=100,
    archive=True,
    snapshot=True,
    history=True,
):
    print(
        f"🚀 Starting sentiment pipeline (method={method}, posts={posts}, comments={comments})"
    )

    # Step 1: Fetch
    raw_data = fetch_all_subreddit_posts_by_dict(
        method=method,
        posts_per_subreddit=posts,
        comment_per_post=comments,
        fetch_buffer=buffer,
    )

    # Step 2: Flatten and prepare for inference
    all_posts = [
        post for cat in raw_data.values() for sub in cat for post in sub["posts"]
    ]
    texts = [
        post["title"]
        + " "
        + post["text"]
        + " "
        + " ".join(c["body"] for c in post["comments"])
        for post in all_posts
    ]

    print(f"🧠 Running inference on {len(texts)} posts...")
    predictions = run_batch_inference(texts)
    for i, post in enumerate(all_posts):
        post["sentiment"] = predictions[i]

    # Step 3: Aggregate
    aggregated = compute_sentiment_average(all_posts)

    # Step 4: Store
    if snapshot:
        save_sentiment_summary(aggregated)
    if history:
        save_sentiment_history(aggregated)
    # TODO: Uncomment when archive functionality is ready
    # if archive:
    #     timestamp = datetime.now(timezone.utc).isoformat()
    #     save_post_archive(all_posts, timestamp=timestamp)

    print("✅ All steps completed.")


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
        posts=args.posts,
        comments=args.comments,
        buffer=args.buffer,
        archive=not args.no_archive,
        snapshot=not args.no_snapshot,
        history=not args.no_history,
    )
