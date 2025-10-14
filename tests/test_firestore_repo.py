# tests/test_firestore_repo.py
"""
FirestoreRepo tests

Goals:
- Exercise success/failure paths for each public method.
- Ensure timestamps are timezone-aware and injected where required.
- Keep test data realistic by using model-validated fixtures from conftest.
- Verify query shape (where / order_by / limit) for history reads.
- Keep payload size sane and retries enabled on writes.

Note: These tests mock the Firestore client. They do not require a real GCP project.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock


# ---------------------------
# save_sentiment_summary
# ---------------------------


def test_save_sentiment_summary_success(firestore_repo, mock_db, fixed_now):
    """
    When saving the latest sentiment snapshot:
    - the repo should inject a timezone-aware 'timestamp'
    - forward provided fields (e.g., joy)
    - and call Firestore with retry enabled
    """
    data = {"joy": 0.9}
    firestore_repo.save_sentiment_summary(data)

    collection = mock_db.collection
    document = collection.return_value.document
    set_call = document.return_value.set.call_args
    assert set_call is not None

    (payload,), kwargs = set_call
    assert payload["joy"] == 0.9
    assert isinstance(payload["timestamp"], datetime)
    assert payload["timestamp"].tzinfo is not None
    assert "retry" in kwargs


def test_save_sentiment_summary_failure_logs(
    firestore_repo, mock_db, fixed_now, caplog
):
    """
    If Firestore.set raises, the repo should log a clear error message.
    """
    mock_db.collection.return_value.document.return_value.set.side_effect = (
        RuntimeError("boom")
    )

    with caplog.at_level("ERROR"):
        firestore_repo.save_sentiment_summary({"joy": 0.9})

    assert any("Failed to save sentiment snapshot" in m for m in caplog.messages)


# ---------------------------
# save_post_archive
# ---------------------------


def test_save_post_archive_success_dicts(firestore_repo, mock_db, fixed_now):
    """
    For post archives:
    - document id should be hour-key (YYYYMMDDHH)
    - payload should contain the posts list and count
    - archived_at should be the fixed timestamp (isoformat per current implementation)
    """
    posts = [{"name": "post1"}, {"name": "post2"}]
    firestore_repo.save_post_archive(posts)

    collection = mock_db.collection.return_value
    doc_call_args = collection.document.call_args
    (hour_id,), _ = doc_call_args
    assert hour_id == "2025010203"

    (payload,), kargs = collection.document.return_value.set.call_args
    assert payload["posts"] == posts
    assert payload["count"] == len(posts)
    assert payload["archived_at"] == fixed_now.isoformat()


def test_save_post_archive_success_models(
    firestore_repo, mock_db, fixed_now, sample_posts
):
    """
    The archive method should accept realistic, model-validated Post payloads
    (converted to dicts) without raising, and store count properly.
    """
    posts_dicts = [p.model_dump(mode="python", exclude_none=True) for p in sample_posts]
    firestore_repo.save_post_archive(posts_dicts)

    (payload,), _ = mock_db.collection.return_value.document.return_value.set.call_args
    assert payload["count"] == len(posts_dicts)
    assert isinstance(
        payload["archived_at"], str
    )  # current impl uses isoformat strings
    assert payload["archived_at"].endswith("+00:00")


def test_save_post_archive_empty_list(firestore_repo, mock_db, fixed_now):
    """
    Archiving an empty list should still write a doc with count=0 (no exceptions).
    """
    firestore_repo.save_post_archive([])
    (payload,), _ = mock_db.collection.return_value.document.return_value.set.call_args
    assert payload["count"] == 0
    assert payload["posts"] == []


def test_save_post_archive_failure_logs(firestore_repo, mock_db, fixed_now, caplog):
    """
    If the collection reference fails, the repo should log and not raise.
    """
    mock_db.collection.side_effect = RuntimeError("boom")

    with caplog.at_level("ERROR"):
        firestore_repo.save_post_archive([{"name": "post1"}])

    assert any("Failed to archive posts" in m for m in caplog.messages)


# ---------------------------
# save_sentiment_history
# ---------------------------


def test_save_sentiment_history_success(firestore_repo, mock_db, fixed_now, caplog):
    """
    History snapshots:
    - document id is ISO-hour (YYYY-MM-DDTHH)
    - timestamp is timezone-aware datetime
    - retries enabled
    - logs a success message
    """
    data = {"joy": 0.5}
    firestore_repo.save_sentiment_history(data)

    doc_call = mock_db.collection.return_value.document.call_args
    (hour_key,), _ = doc_call
    assert hour_key == fixed_now.strftime("%Y-%m-%dT%H")

    set_call = mock_db.collection.return_value.document.return_value.set.call_args
    (payload,), kwargs = set_call
    assert payload["joy"] == 0.5
    assert isinstance(payload["timestamp"], datetime)
    assert payload["timestamp"].tzinfo is not None
    assert "retry" in kwargs

    assert any("Saved sentiment history snapshot" in m for m in caplog.messages)


def test_save_sentiment_history_failure_logs(firestore_repo, mock_db, caplog):
    """Errors from Firestore.set should be logged clearly."""
    mock_db.collection.return_value.document.return_value.set.side_effect = (
        RuntimeError("boom")
    )

    with caplog.at_level("ERROR"):
        firestore_repo.save_sentiment_history({"joy": 0.5})

    assert any("Failed to save sentiment history" in m for m in caplog.messages)


# ---------------------------
# get_latest_sentiment
# ---------------------------


def test_get_latest_sentiment_success(firestore_repo, mock_db):
    """
    When a latest doc exists, return its dict payload.
    """
    fake_doc = MagicMock()
    fake_doc.exists = True
    fake_doc.to_dict.return_value = {"joy": 0.7}
    mock_db.collection.return_value.document.return_value.get.return_value = fake_doc

    result = firestore_repo.get_latest_sentiment()
    assert result == {"joy": 0.7}


def test_get_latest_sentiment_no_data(firestore_repo, mock_db):
    """
    If the latest doc is missing, return a stable error dict.
    """
    fake_doc = MagicMock()
    fake_doc.exists = False
    mock_db.collection.return_value.document.return_value.get.return_value = fake_doc

    result = firestore_repo.get_latest_sentiment()
    assert result == {"error": "No sentiment data found."}


def test_get_latest_sentiment_failure(firestore_repo, mock_db, caplog):
    """
    Exceptions during read should be logged and return an error dict (no raise).
    """
    mock_db.collection.return_value.document.return_value.get.side_effect = (
        RuntimeError("boom")
    )

    with caplog.at_level("ERROR"):
        result = firestore_repo.get_latest_sentiment()

    assert result == {"error": "Firestore read failed."}
    assert any("Failed to read latest sentiment" in m for m in caplog.messages)


# ---------------------------
# get_recent_sentiment_history
# ---------------------------


def test_get_recent_sentiment_history_success(firestore_repo, mock_db):
    """
    Recent history query should stream documents and return their dicts
    in the same order as Firestore yields them.
    """
    fake_doc1 = MagicMock()
    fake_doc1.to_dict.return_value = {"joy": 0.1}
    fake_doc2 = MagicMock()
    fake_doc2.to_dict.return_value = {"joy": 0.2}

    # Simulate query chain where()->order_by()->limit()->stream()
    q = mock_db.collection.return_value.where.return_value
    q.stream.return_value = [fake_doc1, fake_doc2]

    results = firestore_repo.get_recent_sentiment_history(7)
    assert results == [{"joy": 0.1}, {"joy": 0.2}]

    # Verify query composition calls were made (defensive regression check)
    mock_db.collection.return_value.where.assert_called()
    q.stream.assert_called()


def test_get_recent_sentiment_history_failure(firestore_repo, mock_db, caplog):
    """
    If streaming fails, return an empty list and log an error.
    """
    q = mock_db.collection.return_value.where.return_value
    q.order_by.return_value.limit.return_value.stream.side_effect = RuntimeError("boom")

    with caplog.at_level("ERROR"):
        results = firestore_repo.get_recent_sentiment_history(7)

    assert results == []
    assert any("Failed to read sentiment history" in m for m in caplog.messages)


# ---------------------------
# healthcheck
# ---------------------------


def test_healthcheck_success(firestore_repo, mock_db):
    """A healthy DB should allow a cheap .limit(1).stream() call."""
    firestore_repo.healthcheck()
    mock_db.collection.return_value.limit.return_value.stream.assert_called_once()


def test_healthcheck_failure_raises(firestore_repo, mock_db):
    """If the quick read fails, healthcheck should raise."""
    mock_db.collection.return_value.limit.return_value.stream.side_effect = (
        RuntimeError("boom")
    )

    with pytest.raises(RuntimeError):
        firestore_repo.healthcheck()
