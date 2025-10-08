"""Staging environment coverage for the runner pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from app.jobs import runner
from app.models.post import Post, PostComment


class RecordingRepo:
    def __init__(self) -> None:
        self.summary_calls: list[dict[str, Any]] = []
        self.history_calls: list[dict[str, Any]] = []

    def save_sentiment_summary(self, aggregated_sentiment: dict[str, Any]) -> None:
        self.summary_calls.append(aggregated_sentiment)

    def save_sentiment_history(self, aggregated_sentiment: dict[str, Any]) -> None:
        self.history_calls.append(aggregated_sentiment)


def build_post() -> Post:
    return Post(
        post_id="staging-post",
        post_title="Title staging-post",
        post_text="Body staging-post",
        post_comments=[PostComment(body="comment 1")],
        score=100,
        post_comment_count=1,
        post_subreddit="python",
    )


def test_runner_staging_persists_snapshot_and_history(monkeypatch: pytest.MonkeyPatch) -> None:
    post = build_post()
    fetch_mock = Mock(return_value={"dev": [{"name": "python", "posts": [post]}]})
    monkeypatch.setattr(runner, "fetch_all_subreddit_posts_by_dict", fetch_mock)

    inference_mock = Mock(return_value=[{"joy": 0.8, "sadness": 0.1, "anger": 0.1, "fear": 0.0, "love": 0.0, "surprise": 0.0}])
    monkeypatch.setattr(runner, "run_batch_inference", inference_mock)

    repo = RecordingRepo()
    monkeypatch.setattr(runner, "default_repo", lambda: repo)

    bucket_mock = Mock()
    monkeypatch.setattr(runner, "default_bucket_repo", lambda: bucket_mock)

    monkeypatch.setenv("APP_ENV", "staging")

    runner.main(method="new", num_posts=1, num_comments=0, buffer=1, archive=False, snapshot=True, history=True)

    fetch_mock.assert_called_once()
    inference_mock.assert_called_once()
    assert len(repo.summary_calls) == 1, "Staging runs should write the summary when snapshot is enabled"
    assert len(repo.history_calls) == 1, "History persistence should remain active in staging"
    bucket_mock.upload_json.assert_not_called()
