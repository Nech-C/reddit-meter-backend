"""Integration tests for FastAPI sentiment endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app, get_repo


class InMemoryRepo:
    """Simple in-memory repo that records interactions from the API."""

    def __init__(self) -> None:
        self.latest_calls = 0
        self.history_calls: list[int] = []
        self.healthcheck_calls = 0
        self._history_responses: dict[int, list[dict[str, float]]] = {
            1: [{"joy": 0.7, "sadness": 0.2, "anger": 0.1, "timestamp": "2025-07-25T01:40:48+00:00"}],
            7: [{"joy": 0.6, "sadness": 0.25, "anger": 0.15, "timestamp": "2025-07-18T00:00:00+00:00"}],
            31: [{"joy": 0.55, "sadness": 0.3, "anger": 0.15, "timestamp": "2025-06-30T12:00:00+00:00"}],
        }

    def get_latest_sentiment(self) -> dict[str, float]:
        self.latest_calls += 1
        return {"joy": 0.75, "sadness": 0.15, "anger": 0.1}

    def get_recent_sentiment_history(self, num_days: int) -> list[dict[str, float]]:
        self.history_calls.append(num_days)
        return self._history_responses[num_days]

    def healthcheck(self) -> None:
        self.healthcheck_calls += 1


def test_backend_endpoints_work_together(monkeypatch):
    repo = InMemoryRepo()
    monkeypatch.setitem(app.dependency_overrides, get_repo, lambda: repo)

    try:
        with TestClient(app) as client:
            current_resp = client.get("/sentiment/current")
            assert current_resp.status_code == 200
            assert current_resp.json() == {"joy": 0.75, "sadness": 0.15, "anger": 0.1}
            # Security headers are added via middleware.
            assert current_resp.headers["x-content-type-options"].lower() == "nosniff"
            assert current_resp.headers["strict-transport-security"].startswith("max-age=")

            day_resp = client.get("/sentiment/day")
            assert day_resp.status_code == 200
            assert day_resp.json() == repo._history_responses[1]

            week_resp = client.get("/sentiment/week")
            assert week_resp.status_code == 200
            assert week_resp.json() == repo._history_responses[7]

            month_resp = client.get("/sentiment/month")
            assert month_resp.status_code == 200
            assert month_resp.json() == repo._history_responses[31]

            warmup_resp = client.get("/_ah/warmup")
            assert warmup_resp.status_code == 200
            assert warmup_resp.json() == {"status": "ok"}
    finally:
        app.dependency_overrides.clear()

    assert repo.latest_calls == 1
    assert repo.history_calls == [1, 7, 31]
    assert repo.healthcheck_calls == 1
