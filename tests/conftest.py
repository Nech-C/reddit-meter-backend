# tests/conftest.py
"""
Shared pytest fixtures for the reddit-meter backend tests.

This file centralizes commonly used test data so individual tests stay lean,
and ensures data going through the system is as close to production as possible
(by validating via Pydantic models).
"""

import pytest
from fastapi.testclient import TestClient
from slowapi import Limiter
from unittest.mock import MagicMock
from datetime import datetime, timezone, timedelta

from app.storage.firestore import FirestoreRepo
from app.api import main
from app.models.post import (
    Post,
    PostComment,
    Sentiment,
    TopSentimentContributor,
    SentimentSummary,
)


# ---------------------------
# Settings & DB Mocks
# ---------------------------


class DummyStorageSettings:
    CURRENT_SENTIMENT_COLLECTION_NAME = "sentiment_current"
    SENTIMENT_HISTORY_COLLECTION_NAME = "sentiment_history"
    POST_ARCHIVE_COLLECTION_NAME = "post_archive"
    FIRESTORE_DATABASE_ID = "fake-db"
    GOOGLE_BUCKET_NAME = "test-bucket"


@pytest.fixture
def mock_db():
    """
    Mocked Firestore client with a minimal shape to support the repo calls.
    Chainable mocks (collection -> where -> order_by -> limit -> stream) are set
    so tests can assert query composition or override returns as needed.
    """
    db = MagicMock(name="firestore.Client")

    # Base mocks
    collection = MagicMock(name="CollectionRef")
    document = MagicMock(name="DocumentRef")
    query = MagicMock(name="QueryRef")

    # Default chain for history queries: where -> order_by -> limit -> stream
    db.collection.return_value = collection
    collection.document.return_value = document
    collection.where.return_value = query
    query.order_by.return_value = query
    query.limit.return_value = query
    query.stream.return_value = []  # override in tests when needed

    return db


@pytest.fixture
def firestore_repo(mock_db):
    """FirestoreRepo wired to the mocked client and dummy settings."""
    return FirestoreRepo(settings=DummyStorageSettings(), db=mock_db)


@pytest.fixture
def fixed_now(monkeypatch):
    """
    Provide a fixed, timezone-aware 'now' for deterministic tests and patch
    app.storage.firestore.datetime.now(tz=...) to return it.
    """
    fake_now = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "app.storage.firestore.datetime",
        type("dtmod", (), {"now": lambda tz=None: fake_now}),
    )
    return fake_now


# ---------------------------
# Model-validated test data
# ---------------------------


def _make_post(
    *,
    post_id: str,
    title: str = "Title",
    text: str | None = "Body text",
    score: int = 10,
    subreddit: str = "r/test",
    created_ts: datetime | None = None,
    comment_count: int = 1,
    comments: list[PostComment] | None = None,
    sentiment: Sentiment | None = None,
) -> Post:
    """Helper to build a valid Post model instance for tests."""
    created_ts = created_ts or datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc)
    comments = comments or [
        PostComment(body="top comment", author="alice", score=7, created_utc=created_ts)
    ]
    sentiment = sentiment or Sentiment(
        joy=1.0, sadness=0.0, anger=0.0, fear=0.0, love=0.0, surprise=0.0
    )

    return Post(
        post_id=post_id,
        post_url=f"https://reddit.com/{post_id}",
        post_title=title,
        post_text=text,
        post_created_ts=created_ts,
        post_score=score,
        post_comment_count=comment_count,
        post_comments=comments,
        post_subreddit=subreddit,
        sentiment=sentiment,
    )


@pytest.fixture
def sample_posts() -> list[Post]:
    """
    A list of valid Post models, with varying scores and sentiments.
    These are model-validated (closer to real inputs) and can be reused across tests.
    """
    base = datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc)
    return [
        _make_post(
            post_id="p2",
            score=40,
            created_ts=base + timedelta(minutes=1),
            sentiment=Sentiment(
                joy=0.8, sadness=0.1, anger=0.05, fear=0.02, love=0.02, surprise=0.01
            ),
        ),
        _make_post(post_id="p1", score=50, created_ts=base),
        _make_post(
            post_id="p3",
            score=30,
            created_ts=base + timedelta(minutes=2),
            sentiment=Sentiment(
                joy=0.6, sadness=0.2, anger=0.1, fear=0.05, love=0.03, surprise=0.02
            ),
        ),
        _make_post(post_id="p4", score=20, created_ts=base + timedelta(minutes=3)),
        _make_post(post_id="p5", score=10, created_ts=base + timedelta(minutes=4)),
    ]


@pytest.fixture
def sample_summary(sample_posts: list[Post]) -> SentimentSummary:
    """
    A minimal SentimentSummary with top contributors referencing valid Post models.
    Useful when tests want to pass structured data through repo boundaries.
    """
    top = [
        TopSentimentContributor(emotion="joy", top_posts=sample_posts[:3]),
    ]
    # SentimentSummary requires the six emotions plus top_contributors
    return SentimentSummary(
        joy=0.5,
        sadness=0.1,
        anger=0.1,
        fear=0.1,
        love=0.1,
        surprise=0.1,
        top_contributors=top,
    )


# ---------------------------
# FastAPI test client (unchanged)
# ---------------------------


@pytest.fixture
def client(monkeypatch):
    """
    Provide a TestClient for API tests (rate limiter key customized for tests).
    Firestore dependency is overridden with a MagicMock for endpoint-level tests.
    """
    main.app.state.limiter = Limiter(
        key_func=lambda r: r.headers.get("X-Test-Id", "testclient")
    )

    fake_repo = MagicMock(spec=FirestoreRepo)
    main.app.dependency_overrides[main.get_repo] = lambda: fake_repo

    with TestClient(main.app) as test_client:
        yield test_client, fake_repo

    main.app.dependency_overrides.clear()
