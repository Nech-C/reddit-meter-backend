"""Tests for the Reddit fetching utilities."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from app import constants
from app.config import RedditSettings
from app.models.post import Post
from app.reddit.fetch import RedditFetcher


class DummyComment:
    def __init__(self, body: str | None, author: str | None, score: int | None, created_utc: float | None):
        self.body = body
        self.author = author
        self.score = score
        self.created_utc = created_utc


class DummyComments(list):
    def replace_more(self, limit: int):
        self.limit_called = limit


class DummySubmission:
    def __init__(
        self,
        submission_id: str,
        title: str,
        body: str,
        created_utc: float,
        permalink: str,
        score: int,
        num_comments: int,
        comments: Iterable[DummyComment],
    ) -> None:
        self.id = submission_id
        self.title = title
        self.selftext = body
        self.created_utc = created_utc
        self.permalink = permalink
        self.score = score
        self.num_comments = num_comments
        self.comments = DummyComments(comments)


class DummySubreddit:
    def __init__(self, submissions: list[DummySubmission]) -> None:
        self._submissions = submissions

    def hot(self, limit: int):
        yield from self._submissions[:limit]

    def new(self, limit: int):
        yield from self._submissions[:limit]

    def top(self, limit: int):
        yield from self._submissions[:limit]


class DummyReddit:
    def __init__(self, mapping: dict[str, list[DummySubmission]]):
        self._mapping = mapping

    def subreddit(self, name: str) -> DummySubreddit:
        return DummySubreddit(self._mapping[name])


def _settings(tmp_path: Path | None = None) -> RedditSettings:
    subreddit_path = tmp_path if tmp_path else Path("test_subreddits.json")
    return RedditSettings(
        CLIENT_ID="client",
        CLIENT_SECRET="secret",
        PASSWORD="password",
        USER_AGENT="agent",
        USERNAME="user",
        SUBREDDIT_JSON_PATH=str(subreddit_path),
    )


def _submission_with_comments(submission_id: str, comments: list[DummyComment]) -> DummySubmission:
    now = datetime.now(constants.TIMEZONE)
    return DummySubmission(
        submission_id=submission_id,
        title="Interesting discussion",
        body="Detailed body text",
        created_utc=(now - timedelta(hours=1)).timestamp(),
        permalink=f"/r/test/{submission_id}",
        score=42,
        num_comments=len(comments),
        comments=comments,
    )


def test_fetch_subreddit_posts_filters_invalid_comments():
    valid_comment = DummyComment("Great post!", "regular_user", 10, None)
    automod_comment = DummyComment("", "AutoModerator", 5, None)
    blank_comment = DummyComment("   ", "someone", 3, None)
    another_valid = DummyComment("Thanks for sharing", None, -2, None)

    submission = _submission_with_comments(
        "abc123",
        [valid_comment, automod_comment, blank_comment, another_valid],
    )

    reddit_client = DummyReddit({"python": [submission]})
    settings = _settings()
    fetcher = RedditFetcher(settings=settings, reddit_client=reddit_client)

    posts = fetcher.fetch_subreddit_posts(
        subreddit_name="python",
        required_posts=1,
        comment_limit=2,
        fetch_buffer=5,
    )

    assert len(posts) == 1
    assert isinstance(posts[0], Post)
    assert len(posts[0].post_comments) == 2
    assert posts[0].post_comments[0].body == "Great post!"
    # score should be coerced to non-negative
    assert posts[0].post_comments[1].score == 0


def test_fetch_all_subreddit_posts_by_dict_returns_structure(monkeypatch):
    comment = DummyComment("Nice!", "user1", 2, None)
    submission_one = _submission_with_comments("sub1", [comment])
    submission_two = _submission_with_comments("sub2", [comment])

    reddit_client = DummyReddit({
        "python": [submission_one],
        "golang": [submission_two],
    })

    settings = _settings()
    fetcher = RedditFetcher(settings=settings, reddit_client=reddit_client)

    monkeypatch.setattr("app.reddit.fetch.time.sleep", lambda _: None)

    result = fetcher.fetch_all_subreddit_posts_by_dict(
        subreddit_mapping={"tech": ["python", "golang"]},
        posts_per_subreddit=1,
        comment_per_post=1,
        fetch_buffer=2,
    )

    assert set(result.keys()) == {"tech"}
    assert len(result["tech"]) == 2
    assert {entry["name"] for entry in result["tech"]} == {"python", "golang"}
    assert all(entry["posts"] for entry in result["tech"])
