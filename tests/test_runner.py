"""Integration tests for the sentiment ingestion runner."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import Mock

import pytest

from app.jobs import runner
from app.models.post import Post, PostComment, Sentiment
from app import constants


class DummyRepo:
    def __init__(self) -> None:
        self.summary_calls: list[dict[str, Any]] = []
        self.history_calls: list[dict[str, Any]] = []

    def save_sentiment_summary(self, aggregated_sentiment: dict[str, Any]) -> None:
        self.summary_calls.append(aggregated_sentiment)

    def save_sentiment_history(self, aggregated_sentiment: dict[str, Any]) -> None:
        self.history_calls.append(aggregated_sentiment)


class DummyBucket:
    def __init__(self) -> None:
        self.uploads: list[dict[str, Any]] = []

    def upload_json(self, json_data: Any, blob_name: str, bucket_name: str | None = None) -> None:
        self.uploads.append(
            {
                "json_data": json_data,
                "blob_name": blob_name,
                "bucket_name": bucket_name,
            }
        )


def _build_post(post_id: str, subreddit: str, score: int) -> Post:
    return Post(
        post_id=post_id,
        post_title=f"Title {post_id}",
        post_text=f"Body text {post_id}",
        post_comments=[PostComment(body="A helpful comment"), PostComment(body="Another insight")],
        score=score,
        post_comment_count=2,
        post_subreddit=subreddit,
    )


def test_runner_main_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    posts = [_build_post("p1", "python", 50), _build_post("p2", "learnpython", 30)]
    fetch_payload = {"programming": [{"name": "python", "posts": posts}]}

    fetch_mock = Mock(return_value=fetch_payload)
    monkeypatch.setattr(runner, "fetch_all_subreddit_posts_by_dict", fetch_mock)

    predictions = [
        {"joy": 0.7, "sadness": 0.1, "anger": 0.1, "fear": 0.05, "love": 0.05, "surprise": 0.0},
        {"joy": 0.2, "sadness": 0.5, "anger": 0.2, "fear": 0.05, "love": 0.05, "surprise": 0.0},
    ]
    inference_mock = Mock(return_value=predictions)
    monkeypatch.setattr(runner, "run_batch_inference", inference_mock)

    repo = DummyRepo()
    monkeypatch.setattr(runner, "default_repo", lambda: repo)

    bucket = DummyBucket()
    monkeypatch.setattr(runner, "default_bucket_repo", lambda: bucket)

    monkeypatch.setenv("APP_ENV", "production")

    runner.main(
        method="hot",
        num_posts=2,
        num_comments=2,
        buffer=5,
        archive=True,
        snapshot=True,
        history=True,
    )

    fetch_mock.assert_called_once_with(
        method="hot", posts_per_subreddit=2, comment_per_post=2, fetch_buffer=5
    )

    assert inference_mock.call_count == 1
    inference_input = inference_mock.call_args.args[0]
    assert isinstance(inference_input, list)
    assert len(inference_input) == len(posts)
    assert all(isinstance(text, str) and "TITLE:" in text for text in inference_input)

    # Posts should be mutated with sentiment data and consistent timestamps.
    processing_times = {post.processing_timestamp for post in posts}
    assert len(processing_times) == 1
    for post, prediction in zip(posts, predictions, strict=True):
        assert isinstance(post.sentiment, Sentiment)
        assert post.sentiment.model_dump(mode="python") == pytest.approx(prediction, rel=1e-6)
        assert post.sentiment_analysis_model == constants.DEFAULT_SENTIMENT_SOURCE
        assert post.post_subreddit in {"python", "learnpython"}

    assert repo.summary_calls, "Sentiment snapshot should be saved"
    assert repo.history_calls, "Sentiment history should be saved"
    aggregated_summary = repo.summary_calls[0]
    aggregated_history = repo.history_calls[0]
    assert aggregated_summary == aggregated_history
    for key in ("joy", "sadness", "anger", "fear", "love", "surprise"):
        assert key in aggregated_summary
        assert 0.0 <= aggregated_summary[key] <= 1.0
    assert aggregated_summary["_top_contributor"], "Top contributor metadata should be populated"

    assert bucket.uploads, "Archived posts should be uploaded to the bucket"
    archive_payload = bucket.uploads[0]
    assert archive_payload["blob_name"] == posts[0].processing_timestamp.isoformat()
    serialized_posts = archive_payload["json_data"]
    assert isinstance(serialized_posts, list)
    assert len(serialized_posts) == len(posts)
    for serialized in serialized_posts:
        assert serialized["sentiment"]["joy"] >= 0.0
        assert serialized["subreddit"] in {"python", "learnpython"}
        serialized_ts = serialized["processing_timestamp"].replace("Z", "+00:00")
        assert datetime.fromisoformat(serialized_ts) == posts[0].processing_timestamp

    # The runner should not mutate the predictions list that was supplied by the inference mock.
    assert predictions[0]["joy"] == pytest.approx(0.7)
    assert predictions[1]["sadness"] == pytest.approx(0.5)
