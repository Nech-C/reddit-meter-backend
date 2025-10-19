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
from app.storage.bigquery import BigQueryRepo
from app.api import main
from app.models.post import (
    Post,
    PostComment,
    Sentiment,
    TopSentimentContributor,
    SentimentSummary,
)
from app.constants import TIMEZONE


# ---------------------------
# Settings & DB Mocks
# ---------------------------
def get_constant_datetime_fn():
    fixed = datetime(2025, 10, 19, 12, 0, 0, tzinfo=TIMEZONE)

    def _constant_datetime_fn(tz=None):
        if tz is not None:
            return fixed.astimezone(tz)
        return fixed

    return _constant_datetime_fn


@pytest.fixture
def get_constant_datetime():
    return get_constant_datetime_fn()()


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


class DummyBigQuerySettings:
    bq_dataset = "sentiment_dataset"
    bq_global_sentiment_history_table = "sentiment_table"
    bq_global_sentiment_history_limit = "sentiemnt_limit"
    retry = "sentiment_retry"


@pytest.fixture
def mock_bq_client():
    client = MagicMock(name="bigquery.client")

    job = MagicMock(name="QueryJob")
    client.query.return_value = job

    results = MagicMock(name="RowIterator")
    job.results.return_value = results

    return client


@pytest.fixture
def bigquery_repo(mock_bq_client):
    return BigQueryRepo(
        settings=DummyBigQuerySettings,
        client=mock_bq_client,
        now_fn=get_constant_datetime_fn(),
    )


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
    contribution: float = 0,
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
        contribution=contribution,
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


@pytest.fixture
def legacy_output() -> dict:
    return {
        "sadness": 0.143517026257866,
        "joy": 0.393270469837964,
        "love": 0.0225529792857112,
        "updatedAt": datetime.fromisoformat("2025-10-15T01:42:35.833000+00:00"),
        "surprise": 0.0339783017073778,
        "fear": 0.084910188367739,
        "anger": 0.321771034543342,
        "timestamp": datetime.fromisoformat("2025-10-15T01:42:30.861835+00:00"),
        "_top_contributor": {
            "surprise": [
                {
                    "processing_timestamp": "2025-10-15T01:42:19.868678Z",
                    "created": "2025-10-14T18:46:20Z",
                    "subreddit": "mildlyinteresting",
                    "text": "",
                    "contribution": 0.00273763368796457,
                    "id": "1o6o1gc",
                    "url": "https://reddit.com/r/mildlyinteresting/comments/1o6o1gc/arnold_advertising_a_cheap_drill_at_a_discount/",
                    "sentiment_source_model": "bert",
                    "title": "Arnold advertising a cheap drill at a discount store in Germany",
                    "sentiment": {
                        "surprise": 0.924781799316406,
                        "sadness": 0.00484359031543136,
                        "joy": 0.0575864464044571,
                        "anger": 0.00688220653682947,
                        "fear": 0.00384777830913663,
                        "love": 0.00205818004906178,
                    },
                    "comments": [
                        {
                            "author": "crysal0",
                            "created_utc": 1760467735,
                            "score": 7076,
                            "body": "They may be cheap but they are very decent, have had one for 4 years and no issues for the daily task",
                        },
                        {
                            "author": "Aromatic_Fail_1722",
                            "created_utc": 1760467733,
                            "score": 2120,
                            "body": "The same campaign runs here in Belgium (and Holland) too. Everyone's mighty impressed with how they managed to get Ahnold.",
                        },
                        {
                            "author": "elferrydavid",
                            "created_utc": 1760467987,
                            "score": 969,
                            "body": "Parkside superiority!",
                        },
                        {
                            "author": "xondk",
                            "created_utc": 1760468251,
                            "score": 1012,
                            "body": 'I get the idea behind saying it is "cheap" but parkside is generally decent quality, yeah nothing special, but there are a whole host of stuff that is a lot lot worse then parkside.',
                        },
                        {
                            "author": "HugoZHackenbush2",
                            "created_utc": 1760468106,
                            "score": 436,
                            "body": "If you want to buy the product, it's in Aisle B, back..",
                        },
                    ],
                    "num_comments": 1339,
                    "score": 18607,
                },
                {
                    "processing_timestamp": "2025-10-15T01:42:19.868678Z",
                    "created": "2025-10-14T00:43:34Z",
                    "subreddit": "mildlyinteresting",
                    "text": "",
                    "contribution": 0.002565004365351,
                    "id": "1o61e7t",
                    "url": "https://reddit.com/r/mildlyinteresting/comments/1o61e7t/on_japans_bullet_train_the_mens_urinal_door_has_a/",
                    "sentiment_source_model": "bert",
                    "title": "On Japan's bullet train the men's urinal door has a see-through window...",
                    "sentiment": {
                        "surprise": 0.62286102771759,
                        "sadness": 0.0019791501108557,
                        "joy": 0.00202179607003927,
                        "anger": 0.00161215313710272,
                        "fear": 0.370436698198319,
                        "love": 0.00108915090095252,
                    },
                    "comments": [
                        {
                            "author": "Latranis",
                            "created_utc": 1760408553,
                            "score": 4146,
                            "body": "There's a mall in Santa Fe NM where there was apparently a rash of people having sex in the bathroom stalls. Their solution was to take all the doors off the stalls. You haven't had a weird day until you've made eye contact with three consecutive pooping men lined up in a row. In 2001, I had a weird day.",
                        },
                        {
                            "author": "PapaOoMaoMao",
                            "created_utc": 1760403505,
                            "score": 6992,
                            "body": "Went to a tourist bus stop bathroom in Toba. There was no door on the trough room. Anyone walking by could see everything. It was very weird.",
                        },
                        {
                            "author": "froghumper66",
                            "created_utc": 1760403097,
                            "score": 5204,
                            "body": "As long as you’re not one of those weirdo’s that pulls their pants down to piss in the urinal",
                        },
                        {
                            "author": "senorbozz",
                            "created_utc": 1760404371,
                            "score": 2460,
                            "body": "Those are made exclusively by the Swedish company ICUP.",
                        },
                        {
                            "author": "Cptbeeeee",
                            "created_utc": 1760407511,
                            "score": 528,
                            "body": "Don't worry. Your junk is always pixelated in japan",
                        },
                    ],
                    "num_comments": 1214,
                    "score": 30530,
                },
                {
                    "processing_timestamp": "2025-10-15T01:42:19.868678Z",
                    "created": "2025-10-14T11:54:55Z",
                    "subreddit": "mildlyinteresting",
                    "text": "",
                    "contribution": 0.00168886274604596,
                    "id": "1o6dfqh",
                    "url": "https://reddit.com/r/mildlyinteresting/comments/1o6dfqh/found_asterix_obelix_when_i_removed_the_wallpaper/",
                    "sentiment_source_model": "bert",
                    "title": "Found Asterix & Obelix when I removed the wallpaper in our new house.",
                    "sentiment": {
                        "surprise": 0.945324659347534,
                        "sadness": 0.00194967444986105,
                        "joy": 0.00583886355161667,
                        "anger": 0.00146260182373226,
                        "fear": 0.0447047874331474,
                        "love": 0.000719399598892778,
                    },
                    "comments": [
                        {
                            "author": "S_Maja_",
                            "created_utc": 1760444512,
                            "score": 1096,
                            "body": "I've never wanted to steal a wall until now",
                        },
                        {
                            "author": "icantbearsed",
                            "created_utc": 1760443052,
                            "score": 440,
                            "body": "Now you Gettafix it!",
                        },
                        {
                            "author": "archaeo_rex",
                            "created_utc": 1760445619,
                            "score": 251,
                            "body": 'Weird choice to have Goths there, from "Asterix and the Goths"\n\nEven found the exact page where the four Germanics were taken from\n\n[https://dn790008.ca.archive.org/0/items/Asterixcompleteset/Asterix/03-%20Asterix%20and%20the%20Goths.pdf#page=21&zoom=250,-29,18](https://dn790008.ca.archive.org/0/items/Asterixcompleteset/Asterix/03-%20Asterix%20and%20the%20Goths.pdf#page=21&zoom=250,-29,18)',
                        },
                        {
                            "author": "lkap28",
                            "created_utc": 1760447106,
                            "score": 118,
                            "body": "It is my dream to remove wallpaper and find something amazing underneath!! I always make a point of doodling on walls before I cover them up, so I can at least pass that moment on to someone. Is there a date? :)",
                        },
                        {
                            "author": "Muffinshire",
                            "created_utc": 1760447731,
                            "score": 29,
                            "body": "By Toutatis!",
                        },
                    ],
                    "num_comments": 106,
                    "score": 8723,
                },
            ],
        },
    }
