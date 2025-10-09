"""Tests covering staging environment behaviour for the FastAPI backend."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app, get_repo


class StagingRepo:
    def __init__(self) -> None:
        self.latest_calls = 0

    def get_latest_sentiment(self) -> dict[str, float]:
        self.latest_calls += 1
        return {"joy": 0.42, "sadness": 0.33, "anger": 0.25}

    def get_recent_sentiment_history(self, num_days: int) -> list[dict[str, float]]:
        return []

    def healthcheck(self) -> None:
        pass


def test_backend_handles_staging_env(monkeypatch):
    monkeypatch.setenv("APP_ENV", "staging")
    repo = StagingRepo()
    monkeypatch.setitem(app.dependency_overrides, get_repo, lambda: repo)

    try:
        with TestClient(app) as client:
            response = client.get("/sentiment/current")
            assert response.status_code == 200
            assert response.json() == {"joy": 0.42, "sadness": 0.33, "anger": 0.25}
            assert response.headers["x-frame-options"].upper() == "SAMEORIGIN"
    finally:
        app.dependency_overrides.clear()

    assert repo.latest_calls == 1
