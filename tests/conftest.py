# File: tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VER = f"python{sys.version_info.major}.{sys.version_info.minor}"
VENVP = ROOT / ".venv" / "lib" / VER / "site-packages"
if VENVP.exists():
    sys.path.insert(0, str(VENVP))

import pytest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.storage.firestore import FirestoreRepo


@pytest.fixture
def sample_posts():
    # Five valid posts with six-emotion sentiment dicts to exceed top_contributors threshold,
    # plus one with score=0 (ignored), and two invalid ones.
    posts = [
        {  # highest weight
            "sentiment": {
                "joy": 1.0,
                "sadness": 0.0,
                "anger": 0.0,
                "fear": 0.0,
                "love": 0.0,
                "surprise": 0.0,
            },
            "score": 50,
            "id": "p1",
        },
        {  # second highest
            "sentiment": {
                "joy": 0.8,
                "sadness": 0.1,
                "anger": 0.05,
                "fear": 0.02,
                "love": 0.02,
                "surprise": 0.01,
            },
            "score": 40,
            "id": "p2",
        },
        {  # third highest
            "sentiment": {
                "joy": 0.6,
                "sadness": 0.2,
                "anger": 0.1,
                "fear": 0.05,
                "love": 0.03,
                "surprise": 0.02,
            },
            "score": 30,
            "id": "p3",
        },
        {  # fourth highest: will trigger pop for top_contributors
            "sentiment": {
                "joy": 0.4,
                "sadness": 0.3,
                "anger": 0.1,
                "fear": 0.1,
                "love": 0.05,
                "surprise": 0.05,
            },
            "score": 20,
            "id": "p4",
        },
        {  # fifth: lower weight
            "sentiment": {
                "joy": 0.2,
                "sadness": 0.3,
                "anger": 0.2,
                "fear": 0.2,
                "love": 0.05,
                "surprise": 0.05,
            },
            "score": 10,
            "id": "p5",
        },
        {  # zero score => ignored
            "sentiment": {
                "joy": 0.3,
                "sadness": 0.3,
                "anger": 0.2,
                "fear": 0.1,
                "love": 0.05,
                "surprise": 0.05,
            },
            "score": 0,
            "id": "p6",
        },
        {  # invalid: missing sentiment
            "score": 8,
            "id": "invalid1",
        },
        {  # invalid: missing score
            "sentiment": {
                "joy": 0.5,
                "sadness": 0.5,
                "anger": 0.0,
                "fear": 0.0,
                "love": 0.0,
                "surprise": 0.0,
            },
            "id": "invalid2",
        },
    ]
    return posts


class DummyStorageSettings:
    CURRENT_SENTIMENT_COLLECTION_NAME = "sentiment_current"
    SENTIMENT_HISTORY_COLLECTION_NAME = "sentiment_history"
    POST_ARCHIVE_COLLECTION_NAME = "post_archive"
    FIRESTORE_DATABASE_ID = "fake-db"
    GOOGLE_BUCKET_NAME = "test-bucket"


@pytest.fixture
def mock_db():
    db = MagicMock(name="firestore.Client")
    collection = MagicMock(name="CollectionRef")
    document = MagicMock(name="DocumentRef")

    db.collection.return_value = collection
    collection.document.return_value = document

    return db


@pytest.fixture
def firestore_repo(mock_db):
    return FirestoreRepo(settings=DummyStorageSettings(), db=mock_db)


@pytest.fixture
def client(monkeypatch):
    """
    Fixture that yields a TestClient with FirestoreRepo dependency overridden.
    All API routes that use Depends(get_repo) will receive a MagicMock instead.
    """
    from app.api import main

    fake_repo = MagicMock(spec=FirestoreRepo)

    # Override FastAPI dependency injection
    main.app.dependency_overrides[main.get_repo] = lambda: fake_repo

    with TestClient(main.app) as test_client:
        # expose both client and fake_repo
        yield test_client, fake_repo

    # cleanup overrides after test
    main.app.dependency_overrides.clear()
