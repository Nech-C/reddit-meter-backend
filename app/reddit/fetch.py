# File: app/reddit/fetch.py
import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv
import praw

env_name = os.getenv("APP_ENV", "dev")
load_dotenv(f".env.{env_name}")

SUBREDDIT_JSON_PATH = os.getenv("SUBREDDIT_JSON_PATH")
DEFAULT_SUBBREDDITS_BY_CATEGORY = json.load(open(SUBREDDIT_JSON_PATH, "r"))
reddit = None


def initialize():
    global reddit
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        password=os.getenv("REDDIT_PASSWORD"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
        username=os.getenv("REDDIT_USERNAME"),
        ratelimit_seconds=int(os.getenv("REDDIT_RATELIMIT_SECONDS", "600")),
    )

    # check if it's initialized
    if reddit.user.me() != os.getenv("REDDIT_USERNAME"):
        raise ValueError(
            "Reddit API initialization failed. Check your credentials in .env file."
        )
    print(f"Logged in as {reddit.user.me()}")


initialize()


def fetch_subreddit_posts(
    subreddit_name: str,
    method: str = "hot",
    required_posts: int = 15,
    comment_limit: int = 5,
    fetch_buffer: int = 100,
) -> list:
    """
    Fetch up to `required_posts` valid posts, each with at least `comment_limit` valid comments.
    Filters out image-based posts and comments.

    Args:
        subreddit_name (str): Subreddit to fetch from.
        method (str): 'hot', 'new', or 'top'. Defaults to 'hot'.
        required_posts (int): Number of usable posts to return. defaults to 15.
        comment_limit (int): Minimum number of valid comments per post. Defaults to 5.
        fetch_buffer (int): How many posts to sample total. Defaults to 100.

    Returns:
        list: List of filtered post dictionaries.
    """
    if method not in ["hot", "new", "top"]:
        raise ValueError("Method must be one of 'hot', 'new', or 'top'.")

    subreddit = reddit.subreddit(subreddit_name)
    fetch_method = getattr(subreddit, method)
    results = []
    collected = 0

    for submission in fetch_method(limit=fetch_buffer):
        # Filter out image/empty posts
        if not submission.selftext.strip() and not submission.title.strip():
            continue

        try:
            print(f"Processing submission: {submission.id} - {submission.title}")
            submission.comments.replace_more(limit=5)

            valid_comments = []
            for comment in submission.comments:
                if not comment.body or not comment.body.strip():
                    continue
                valid_comments.append(
                    {
                        "body": comment.body,
                        "author": (
                            str(comment.author) if comment.author else "[deleted]"
                        ),
                        "score": comment.score,
                    }
                )
                if len(valid_comments) >= comment_limit:
                    break

            # Only keep the post if enough valid comments
            if len(valid_comments) < comment_limit:
                continue

            post_data = {
                "id": submission.id,
                "title": submission.title,
                "text": submission.selftext,
                "url": submission.url,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "created_utc": submission.created_utc,
                "created": datetime.utcfromtimestamp(
                    submission.created_utc
                ).isoformat(),
                "comments": valid_comments,
                "subreddit": subreddit_name,
            }

            results.append(post_data)
            collected += 1
            if collected >= required_posts:
                break

        except Exception as e:
            print(f"Error processing submission {submission.id}: {e}")

    if collected < required_posts:
        print(
            f"Warning: Only collected {collected}/{required_posts} posts after fetching {fetch_buffer}"
        )

    return results


def fetch_all_subreddit_posts_by_dict(
    sub_dict: dict = DEFAULT_SUBBREDDITS_BY_CATEGORY,
    method: str = "hot",
    posts_per_subreddit: int = 15,
    comment_per_post: int = 5,
    fetch_buffer: int = 100,
) -> dict:
    """
    Fetch posts by subreddit dictionary.
    Args:
        sub_dict (dict): Dictionary of subreddit categories and their subreddits. Defaults to the loaded JSON from `subreddits.json`.
            {
                "subreddit category": [
                    "subreddit1",
                    "subreddit2",
                    ...
                ]
            }

        method (str): 'hot', 'new', or 'top'. Defaults to 'hot'.
        posts_per_subreddit (int): Number of usable posts to return per subreddit. Defaults to 15.
        comment_per_post (int): Minimum number of valid comments per post. Defaults to 5.
        fetch_buffer (int): How many posts to sample total per subreddit. Defaults to 100.
    Returns:
        dict: Dictionary of subreddit names to lists of post dictionaries.
    """

    res = {}
    for category, subreddits in sub_dict.items():
        res[category] = []
        for subreddit in subreddits:
            subreddit_dict = {
                "name": subreddit,
                "posts": fetch_subreddit_posts(
                    subreddit_name=subreddit,
                    method=method,
                    required_posts=posts_per_subreddit,
                    comment_limit=comment_per_post,
                    fetch_buffer=fetch_buffer,
                ),
            }
            res[category].append(subreddit_dict)
            print(
                f"Fetched {len(subreddit_dict['posts'])} posts from {subreddit} in category {category}"
            )
        print(
            f"Completed fetching category: {category} with {len(res[category])} subreddits."
        )
    return res
