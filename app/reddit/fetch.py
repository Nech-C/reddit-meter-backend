# File: app/reddit/fetch.py
"""Utilities for fetching Reddit posts and converting them into typed models."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import praw

from app import constants
from app.config import RedditSettings, get_reddit_settings
from app.models.post import Post, PostComment


log = logging.getLogger("reddit.fetch")


class RedditFetcher:
    """Encapsulates the logic for fetching and preparing Reddit posts."""

    def __init__(
        self,
        settings: RedditSettings | None = None,
        reddit_client: praw.Reddit | None = None,
        *,
        subreddit_config_path: str | None = None,
    ) -> None:
        """Create a new :class:`RedditFetcher`.

        Args:
            settings: Optional settings instance. If omitted a cached instance
                from :func:`app.config.get_reddit_settings` will be used.
            reddit_client: Pre-configured PRAW client. When ``None`` a new
                client is created from ``settings``.
            subreddit_config_path: Optional override for the default JSON file
                containing subreddit categories.
        """

        self.settings = settings or get_reddit_settings()
        self._reddit = reddit_client or self._build_reddit_client()
        self._subreddit_config_path = (
            Path(subreddit_config_path)
            if subreddit_config_path
            else Path(self.settings.SUBREDDIT_JSON_PATH)
        )
        self._default_subreddits_by_category = self._load_default_subreddits()

    def _build_reddit_client(self) -> praw.Reddit:
        """Create a new authenticated PRAW client using configured settings."""

        client = praw.Reddit(
            client_id=self.settings.CLIENT_ID,
            client_secret=self.settings.CLIENT_SECRET,
            password=self.settings.PASSWORD,
            user_agent=self.settings.USER_AGENT,
            username=self.settings.USERNAME,
            ratelimit_seconds=self.settings.RATELIMIT_SECONDS,
        )

        authenticated_user = client.user.me()
        if authenticated_user != self.settings.USERNAME:
            raise ValueError(
                "Reddit API initialization failed. Check your credentials in the environment."
            )
        log.info("Logged in as %s", authenticated_user)
        return client

    def _load_default_subreddits(self) -> dict[str, list[str]]:
        """Load the JSON mapping of categories to subreddits."""

        with self._subreddit_config_path.open(encoding="utf-8") as config_file:
            data = json.load(config_file)

        if not isinstance(data, dict):
            raise ValueError(
                "Subreddit configuration must be a JSON object mapping categories to subreddit lists."
            )
        return data

    @property
    def default_subreddits_by_category(self) -> dict[str, list[str]]:
        """Expose the default subreddit configuration."""

        return self._default_subreddits_by_category

    def fetch_subreddit_posts(
        self,
        subreddit_name: str,
        method: str = "hot",
        required_posts: int = 15,
        comment_limit: int = 5,
        fetch_buffer: int = 100,
        max_post_age_days: int = constants.DEFAULT_MAX_POST_AGE_DAYS,
    ) -> list[Post]:
        """Fetch and normalize posts for a single subreddit.

        Args:
            subreddit_name: Name of the subreddit to query.
            method: Listing strategy (``hot``, ``new`` or ``top``).
            required_posts: Number of valid posts to return.
            comment_limit: Minimum number of valid comments per post.
            fetch_buffer: Total number of submissions to inspect.
            max_post_age_days: Maximum age of posts to include.

        Returns:
            A list of validated :class:`~app.models.post.Post` instances.
        """

        if method not in {"hot", "new", "top"}:
            raise ValueError("Method must be one of 'hot', 'new', or 'top'.")

        subreddit = self._reddit.subreddit(subreddit_name)
        listing_method = getattr(subreddit, method)
        normalized_posts: list[Post] = []
        collected_posts = 0
        cutoff_timestamp = (
            datetime.now(constants.TIMEZONE) - timedelta(days=max_post_age_days)
        ).timestamp()

        for submission in listing_method(limit=fetch_buffer):
            if not submission.selftext.strip() and not submission.title.strip():
                continue

            if submission.created_utc < cutoff_timestamp:
                continue

            try:
                log.debug(
                    "Processing submission %s - %s", submission.id, submission.title
                )
                submission.comments.replace_more(limit=5)

                valid_comments: list[PostComment] = []
                for comment in submission.comments:
                    if not comment.body or not comment.body.strip():
                        continue
                    if comment.author == "AutoModerator":
                        continue
                    valid_comments.append(
                        PostComment(
                            body=comment.body,
                            author=(
                                str(comment.author)
                                if comment.author
                                else constants.DEFAULT_COMMENT_AUTHOR_PLACEHOLDER
                            ),
                            score=max(comment.score or 0, 0),
                            created_utc=getattr(comment, "created_utc", None),
                        )
                    )
                    if len(valid_comments) >= comment_limit:
                        break

                if len(valid_comments) < comment_limit:
                    continue

                post_model = Post(
                    post_id=submission.id,
                    post_title=submission.title,
                    post_text=submission.selftext,
                    post_url=f"https://reddit.com{submission.permalink}",
                    score=submission.score,
                    post_comment_count=submission.num_comments,
                    post_created_ts=datetime.fromtimestamp(
                        submission.created_utc, tz=constants.TIMEZONE
                    ),
                    post_comments=valid_comments,
                    post_subreddit=subreddit_name,
                )

                normalized_posts.append(post_model)
                collected_posts += 1
                if collected_posts >= required_posts:
                    break

            except Exception as exc:  # pragma: no cover - defensive logging
                log.exception(
                    "Error processing submission %s: %s", submission.id, exc
                )

        if collected_posts < required_posts:
            log.warning(
                "Only collected %s/%s posts after fetching %s entries from %s",
                collected_posts,
                required_posts,
                fetch_buffer,
                subreddit_name,
            )

        return normalized_posts

    def fetch_all_subreddit_posts_by_dict(
        self,
        subreddit_mapping: Mapping[str, Iterable[str]] | None = None,
        method: str = "hot",
        posts_per_subreddit: int = 15,
        comment_per_post: int = 5,
        fetch_buffer: int = 100,
    ) -> dict[str, list[dict[str, list[Post]]]]:
        """Fetch posts for all subreddits defined by a category mapping.

        Args:
            subreddit_mapping: Mapping of category name to iterable of subreddit names.
                When omitted, :attr:`default_subreddits_by_category` is used.
            method: Listing strategy to apply for each subreddit.
            posts_per_subreddit: Number of valid posts to fetch per subreddit.
            comment_per_post: Minimum number of valid comments per post.
            fetch_buffer: Total number of submissions to inspect per subreddit.

        Returns:
            Dictionary keyed by category containing subreddit data with posts.
        """

        subreddits_by_category = (
            subreddit_mapping if subreddit_mapping is not None else self.default_subreddits_by_category
        )

        aggregated_results: dict[str, list[dict[str, list[Post]]]] = {}
        for category_name, subreddit_names in subreddits_by_category.items():
            aggregated_results[category_name] = []
            for subreddit_name in subreddit_names:
                time.sleep(constants.DEFAULT_FETCH_SLEEP_SECONDS)
                posts = self.fetch_subreddit_posts(
                    subreddit_name=subreddit_name,
                    method=method,
                    required_posts=posts_per_subreddit,
                    comment_limit=comment_per_post,
                    fetch_buffer=fetch_buffer,
                )
                aggregated_results[category_name].append(
                    {"name": subreddit_name, "posts": posts}
                )
                log.info(
                    "Fetched %s posts from %s in category %s",
                    len(posts),
                    subreddit_name,
                    category_name,
                )
            log.info(
                "Completed fetching category %s with %s subreddits.",
                category_name,
                len(aggregated_results[category_name]),
            )
        return aggregated_results


@lru_cache(maxsize=1)
def default_fetcher() -> RedditFetcher:
    """Return a cached :class:`RedditFetcher` instance."""

    return RedditFetcher()


def fetch_subreddit_posts(
    subreddit_name: str,
    method: str = "hot",
    required_posts: int = 15,
    comment_limit: int = 5,
    fetch_buffer: int = 100,
    max_post_age_days: int = constants.DEFAULT_MAX_POST_AGE_DAYS,
) -> list[Post]:
    """Fetch subreddit posts using the shared :class:`RedditFetcher` instance.

    Args mirror :meth:`RedditFetcher.fetch_subreddit_posts`.
    """

    return default_fetcher().fetch_subreddit_posts(
        subreddit_name=subreddit_name,
        method=method,
        required_posts=required_posts,
        comment_limit=comment_limit,
        fetch_buffer=fetch_buffer,
        max_post_age_days=max_post_age_days,
    )


def fetch_all_subreddit_posts_by_dict(
    sub_dict: Mapping[str, Iterable[str]] | None = None,
    method: str = "hot",
    posts_per_subreddit: int = 15,
    comment_per_post: int = 5,
    fetch_buffer: int = 100,
) -> dict[str, list[dict[str, list[Post]]]]:
    """Fetch posts for multiple subreddits using the shared fetcher instance.

    Args mirror :meth:`RedditFetcher.fetch_all_subreddit_posts_by_dict`.
    """

    return default_fetcher().fetch_all_subreddit_posts_by_dict(
        subreddit_mapping=sub_dict,
        method=method,
        posts_per_subreddit=posts_per_subreddit,
        comment_per_post=comment_per_post,
        fetch_buffer=fetch_buffer,
    )
