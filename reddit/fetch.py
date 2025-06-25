import json
import os
from datetime import datetime

from dotenv import load_dotenv
import praw

load_dotenv()

SUBREDDITS_JSON_PATH = "./subreddits.json"
DEFAULT_SUBBREDDITS_BY_CATEGORY = json.load(open(SUBREDDITS_JSON_PATH, "r"))

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
    username=os.getenv("REDDIT_USERNAME")
)


def fetch_subreddit_posts(subreddit_name: str,
                          method: str = 'hot',
                          required_posts: int = 15,
                          comment_limit: int = 5,
                          fetch_buffer: int = 100) -> list:
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
    if method not in ['hot', 'new', 'top']:
        raise ValueError("Method must be one of 'hot', 'new', or 'top'.")

    subreddit = reddit.subreddit(subreddit_name)
    fetch_method = getattr(subreddit, method)
    results = []
    collected = 0

    for submission in fetch_method(limit=fetch_buffer):
        # Filter out image/empty posts
        if (
            submission.url.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")) or
            (hasattr(submission, "post_hint") and submission.post_hint == "image") or
            not submission.selftext.strip()
        ):
            continue

        try:
            submission.comments.replace_more(limit=1)

            valid_comments = []
            for comment in submission.comments:
                if "http" in comment.body and any(ext in comment.body.lower() for ext in [".jpg", ".png", ".gif", ".webp", "i.redd.it", "imgur.com"]):
                    continue
                valid_comments.append({
                    'body': comment.body,
                    'author': str(comment.author) if comment.author else "[deleted]",
                    'score': comment.score
                })
                if len(valid_comments) >= comment_limit:
                    break

            # Only keep the post if enough valid comments
            if len(valid_comments) < comment_limit:
                continue

            post_data = {
                'id': submission.id,
                'title': submission.title,
                'url': submission.url,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'created_utc': submission.created_utc,
                'created': datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                'comments': valid_comments
            }

            results.append(post_data)
            collected += 1
            if collected >= required_posts:
                break

        except Exception as e:
            print(f"Error processing submission {submission.id}: {e}")

    return results


def fetch_all_subreddit_posts_by_dict(sub_dict: dict = DEFAULT_SUBBREDDITS_BY_CATEGORY,
                                      method: str = 'hot',
                                      posts_per_subreddit: int = 15,
                                      comment_per_post: int = 5,
                                      fetch_buffer: int = 100
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
        
        method (str): 'hot', 'new', or 'top'.
    """