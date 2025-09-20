import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock


@pytest.fixture
def fixed_now(monkeypatch):
    fake_now = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "app.storage.firestore.datetime",
        type("dtmod", (), {"now": lambda tz=None: fake_now}),
    )
    return fake_now


def test_save_sentiment_summary_success(firestore_repo, mock_db, monkeypatch):
    fixed_now = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "app.storage.firestore.datetime",
        type("dtmod", (), {"now": lambda tz: fixed_now}),
    )
    data = {"joy": 0.9}
    firestore_repo.save_sentiment_summary(data)

    collection = mock_db.collection
    document = collection.return_value.document
    set_call = document.return_value.set.call_args
    assert set_call is not None

    (payload,), kwargs = set_call
    assert payload["joy"] == 0.9
    assert isinstance(payload["timestamp"], datetime)
    assert "retry" in kwargs


def test_save_sentiment_summary_failure(firestore_repo, mock_db, monkeypatch, caplog):
    mock_db.collection.return_value.document.return_value.set.side_effect = (
        RuntimeError("boom")
    )

    fixed_now = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "app.storage.firestore.datetime",
        type("dtmod", (), {"now": lambda tz: fixed_now}),
    )

    with caplog.at_level("ERROR"):
        firestore_repo.save_sentiment_summary({"joy": 0.9})

    assert any("Failed to save sentiment snapshot" in m for m in caplog.messages)


def test_save_post_archive_success(firestore_repo, mock_db, fixed_now):
    posts = [{"name": "post1"}, {"name": "post2"}]
    firestore_repo.save_post_archive(posts)

    collection = mock_db.collection.return_value
    docu_call_args = collection.document.call_args
    (hour_id,), _ = docu_call_args
    assert hour_id == "2025010203"

    (payload,), kargs = collection.document.return_value.set.call_args
    assert payload["posts"] == posts
    assert payload["count"] == len(posts)
    assert payload["archived_at"] == fixed_now.isoformat()


def test_save_post_archive_failure(firestore_repo, mock_db, fixed_now, caplog):
    posts = [{"name": "post1"}, {"name": "post2"}]
    mock_db.collection.side_effect = RuntimeError("boom")

    with caplog.at_level("ERROR"):
        firestore_repo.save_post_archive(posts)

    assert any("Failed to archive posts" in m for m in caplog.messages)


def test_save_sentiment_history_success(firestore_repo, mock_db, fixed_now, caplog):
    data = {"joy": 0.5}
    firestore_repo.save_sentiment_history(data)

    doc_call = mock_db.collection.return_value.document.call_args
    (hour_key,), _ = doc_call
    assert hour_key == fixed_now.strftime("%Y-%m-%dT%H")

    set_call = mock_db.collection.return_value.document.return_value.set.call_args
    (payload,), kwargs = set_call
    assert payload["joy"] == 0.5
    assert isinstance(payload["timestamp"], datetime)
    assert "retry" in kwargs

    assert any("Saved sentiment history snapshot" in m for m in caplog.messages)


def test_save_sentiment_history_failure(firestore_repo, mock_db, caplog):
    mock_db.collection.return_value.document.return_value.set.side_effect = (
        RuntimeError("boom")
    )

    with caplog.at_level("ERROR"):
        firestore_repo.save_sentiment_history({"joy": 0.5})

    assert any("Failed to save sentiment history" in m for m in caplog.messages)


def test_get_latest_sentiment_success(firestore_repo, mock_db):
    fake_doc = MagicMock()
    fake_doc.exists = True
    fake_doc.to_dict.return_value = {"joy": 0.7}
    mock_db.collection.return_value.document.return_value.get.return_value = fake_doc

    result = firestore_repo.get_latest_sentiment()
    assert result == {"joy": 0.7}


def test_get_latest_sentiment_no_data(firestore_repo, mock_db):
    fake_doc = MagicMock()
    fake_doc.exists = False
    mock_db.collection.return_value.document.return_value.get.return_value = fake_doc

    result = firestore_repo.get_latest_sentiment()
    assert result == {"error": "No sentiment data found."}


def test_get_latest_sentiment_failure(firestore_repo, mock_db, caplog):
    mock_db.collection.return_value.document.return_value.get.side_effect = (
        RuntimeError("boom")
    )

    with caplog.at_level("ERROR"):
        result = firestore_repo.get_latest_sentiment()

    assert result == {"error": "Firestore read failed."}
    assert any("Failed to read latest sentiment" in m for m in caplog.messages)


def test_get_recent_sentiment_history_success(firestore_repo, mock_db):
    fake_doc1 = MagicMock()
    fake_doc1.to_dict.return_value = {"joy": 0.1}
    fake_doc2 = MagicMock()
    fake_doc2.to_dict.return_value = {"joy": 0.2}

    mock_db.collection.return_value.where.return_value.stream.return_value = [
        fake_doc1,
        fake_doc2,
    ]

    results = firestore_repo.get_recent_sentiment_history(7)
    assert results == [{"joy": 0.1}, {"joy": 0.2}]


def test_get_recent_sentiment_history_failure(firestore_repo, mock_db, caplog):
    mock_db.collection.return_value.where.return_value.stream.side_effect = (
        RuntimeError("boom")
    )

    with caplog.at_level("ERROR"):
        results = firestore_repo.get_recent_sentiment_history(7)

    assert results == []
    assert any("Failed to read sentiment history" in m for m in caplog.messages)


def test_healthcheck_success(firestore_repo, mock_db):
    firestore_repo.healthcheck()
    mock_db.collection.return_value.limit.return_value.stream.assert_called_once()


def test_healthcheck_failure(firestore_repo, mock_db):
    mock_db.collection.return_value.limit.return_value.stream.side_effect = (
        RuntimeError("boom")
    )

    with pytest.raises(RuntimeError):
        firestore_repo.healthcheck()
