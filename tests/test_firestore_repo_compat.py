# """
# Compatibility tests for FirestoreRepo read-paths.

# These verify that when APP_OUT is 'legacy' the repo emits the old frontend
# shape regardless of whether Firestore stored the OLD or NEW schema. They also
# verify that when APP_OUT is 'new' the NEW shape is returned.

# We monkeypatch `app.storage.firestore.APP_OUT` per test so other tests are not
# affected (pytest restores after each test).
# """

# from __future__ import annotations

# from datetime import datetime, timezone
# from unittest.mock import MagicMock

# import pytest


# # ---- helpers to craft fake Firestore docs ----

# def _new_post_dict() -> dict:
#     # Minimal but representative "NEW" post shape (already JSON-like)
#     return {
#         "post_id": "p123",
#         "post_url": "https://reddit.com/p123",
#         "post_title": "Hello",
#         "post_text": "full body that should not go to BQ",
#         "post_text_preview": "full body that shoul",  # doesn't matter for FE
#         "post_created_ts": "2025-01-01T00:00:00+00:00",
#         "post_comment_count": 2,
#         "post_score": 50,
#         "post_subreddit": "python",
#         "post_comments": [
#             {"body": "great", "author": "alice", "score": 10, "created_utc": "2025-01-01T00:01:00+00:00"}
#         ],
#         "contribution": 0.42,
#         "sentiment": {
#             "joy": 1.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "love": 0.0, "surprise": 0.0
#         },
#         "processing_timestamp": "2025-01-01T00:02:00+00:00",
#         "sentiment_analysis_model": "bert",
#         "sentiment_model_version": "v1",
#     }


# def _new_summary_doc(ts: datetime, ua: datetime) -> dict:
#     return {
#         "joy": 0.5, "sadness": 0.1, "anger": 0.1, "fear": 0.1, "love": 0.1, "surprise": 0.1,
#         "timestamp": ts,          # repo should serialize to iso string in new/legacy
#         "updated_at": ua,
#         "top_contributors": [
#             {"emotion": "joy", "top_posts": [_new_post_dict()]}
#         ],
#     }


# def _old_summary_doc(ts: datetime, ua: datetime) -> dict:
#     # OLD doc as it existed before (using _top_contributor + camelCase updatedAt)
#     return {
#         "joy": 0.5, "sadness": 0.1, "anger": 0.1, "fear": 0.1, "love": 0.1, "surprise": 0.1,
#         "timestamp": ts,
#         "updatedAt": ua,
#         "_top_contributor": {
#             "joy": [
#                 {
#                     "id": "p999",
#                     "url": "https://reddit.com/p999",
#                     "title": "Legacy Post",
#                     "text": "",
#                     "created": "2025-01-01T00:00:00+00:00",
#                     "num_comments": 3,
#                     "score": 12,
#                     "subreddit": "learnpython",
#                     "comments": [],
#                     "contribution": 0.1,
#                     "sentiment": {
#                         "joy": 0.9, "sadness": 0.05, "anger": 0.0, "fear": 0.03, "love": 0.02, "surprise": 0.0
#                     },
#                     "processing_timestamp": "2025-01-01T00:05:00+00:00",
#                     "sentiment_source_model": "bert",
#                     "sentiment_model_version": "v0",
#                 }
#             ]
#         },
#     }


# # ---- tests ----

# def test_latest_sentiment_legacy_from_NEW_doc(monkeypatch, firestore_repo, mock_db):
#     """When APP_OUT='legacy' and Firestore stores NEW schema, API emits legacy shape."""
#     # toggle legacy output just for this test
#     import app.storage.firestore as fs
#     monkeypatch.setattr(fs, "APP_OUT", "legacy")

#     ts = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
#     ua = datetime(2025, 1, 2, 3, 5, 5, tzinfo=timezone.utc)
#     doc = MagicMock()
#     doc.exists = True
#     doc.to_dict.return_value = _new_summary_doc(ts, ua)
#     mock_db.collection.return_value.document.return_value.get.return_value = doc

#     out = firestore_repo.get_latest_sentiment()

#     # legacy keys
#     assert "_top_contributor" in out
#     assert "updatedAt" in out
#     # and not the new ones
#     assert "top_contributors" not in out
#     assert "updated_at" not in out

#     # timestamps serialized to iso strings
#     assert isinstance(out["timestamp"], str)
#     assert isinstance(out["updatedAt"], str)
#     assert out["timestamp"].endswith("+00:00")
#     assert out["updatedAt"].endswith("+00:00")

#     # nested post mapped to legacy post fields
#     posts = out["_top_contributor"]["joy"]
#     assert isinstance(posts, list) and posts
#     p0 = posts[0]
#     for k in ("id", "url", "title", "text", "created", "num_comments", "score", "subreddit",
#               "contribution", "sentiment", "processing_timestamp", "sentiment_source_model"):
#         assert k in p0


# def test_latest_sentiment_legacy_from_OLD_doc(monkeypatch, firestore_repo, mock_db):
#     """When APP_OUT='legacy' and Firestore stores OLD schema, API still emits legacy shape."""
#     import app.storage.firestore as fs
#     monkeypatch.setattr(fs, "APP_OUT", "legacy")

#     ts = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
#     ua = datetime(2025, 1, 2, 3, 5, 5, tzinfo=timezone.utc)
#     doc = MagicMock()
#     doc.exists = True
#     doc.to_dict.return_value = _old_summary_doc(ts, ua)
#     mock_db.collection.return_value.document.return_value.get.return_value = doc

#     out = firestore_repo.get_latest_sentiment()

#     # it should already be legacy and remain legacy
#     assert "_top_contributor" in out
#     assert "updatedAt" in out
#     assert "top_contributors" not in out
#     assert "updated_at" not in out


# def test_latest_sentiment_NEW_output_from_NEW_doc(monkeypatch, firestore_repo, mock_db):
#     """When APP_OUT='new', API emits new schema."""
#     import app.storage.firestore as fs
#     monkeypatch.setattr(fs, "APP_OUT", "new")

#     ts = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
#     ua = datetime(2025, 1, 2, 3, 5, 5, tzinfo=timezone.utc)
#     doc = MagicMock()
#     doc.exists = True
#     doc.to_dict.return_value = _new_summary_doc(ts, ua)
#     mock_db.collection.return_value.document.return_value.get.return_value = doc

#     out = firestore_repo.get_latest_sentiment()

#     # new keys present
#     assert "top_contributors" in out
#     assert "updated_at" in out
#     # legacy keys absent
#     assert "_top_contributor" not in out
#     assert "updatedAt" not in out

#     # timestamps serialized
#     assert isinstance(out["timestamp"], str)
#     assert isinstance(out["updated_at"], str)


# def test_history_sentiment_legacy_mixed_docs(monkeypatch, firestore_repo, mock_db):
#     """History list converts each item to legacy regardless of input shape."""
#     import app.storage.firestore as fs
#     monkeypatch.setattr(fs, "APP_OUT", "legacy")

#     ts = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
#     ua = datetime(2025, 1, 2, 3, 5, 5, tzinfo=timezone.utc)
#     d1 = MagicMock(); d1.to_dict.return_value = _new_summary_doc(ts, ua)
#     d2 = MagicMock(); d2.to_dict.return_value = _old_summary_doc(ts, ua)

#     q = mock_db.collection.return_value.where.return_value
#     q.stream.return_value = [d1, d2]

#     out = firestore_repo.get_recent_sentiment_history(7)
#     assert len(out) == 2
#     for item in out:
#         assert "_top_contributor" in item
#         assert "top_contributors" not in item
#         assert "updatedAt" in item
#         assert "updated_at" not in item
