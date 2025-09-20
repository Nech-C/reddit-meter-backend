import pytest


def test_read_root(client):
    test_client, _ = client
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}


def test_current_sentiment(client):
    test_client, fake_repo = client
    fake_repo.get_latest_sentiment.return_value = {"joy": 0.6}

    response = test_client.get("/sentiment/current")
    assert response.status_code == 200
    assert response.json() == {"joy": 0.6}


def test_get_past_day_sentiment(client):
    test_client, fake_repo = client
    fake_repo.get_recent_sentiment_history.return_value = [
        {
            "love": 0.02,
            "anger": 0.27,
            "sadness": 0.13,
            "joy": 0.43,
            "timestamp": "2025-07-25T01:40:48.024915+00:00",
            "surprise": 0.05,
        },
        {
            "love": 0.02,
            "anger": 0.27,
            "sadness": 0.13,
            "joy": 0.43,
            "timestamp": "2025-07-25T03:40:48.024915+00:00",
            "surprise": 0.05,
        },
    ]

    response = test_client.get("/sentiment/day")

    assert response.status_code == 200
    assert response.json() == fake_repo.get_recent_sentiment_history.return_value
    fake_repo.get_recent_sentiment_history.assert_called_once_with(1)


def test_get_past_week_sentiment(client):
    test_client, fake_repo = client
    fake_repo.get_recent_sentiment_history.return_value = [
        {
            "love": 0.1,
            "anger": 0.2,
            "sadness": 0.15,
            "joy": 0.4,
            "timestamp": "2025-07-18T01:00:00.000000+00:00",
            "surprise": 0.15,
        },
        {
            "love": 0.12,
            "anger": 0.25,
            "sadness": 0.10,
            "joy": 0.48,
            "timestamp": "2025-07-24T01:00:00.000000+00:00",
            "surprise": 0.05,
        },
    ]

    response = test_client.get("/sentiment/week")
    assert response.status_code == 200
    assert response.json() == fake_repo.get_recent_sentiment_history.return_value
    fake_repo.get_recent_sentiment_history.assert_called_once_with(7)


def test_get_past_month_sentiment(client):
    test_client, fake_repo = client
    fake_repo.get_recent_sentiment_history.return_value = [
        {
            "love": 0.05,
            "anger": 0.3,
            "sadness": 0.1,
            "joy": 0.35,
            "timestamp": "2025-06-25T12:00:00.000000+00:00",
            "surprise": 0.2,
        },
        {
            "love": 0.03,
            "anger": 0.22,
            "sadness": 0.17,
            "joy": 0.38,
            "timestamp": "2025-07-20T18:00:00.000000+00:00",
            "surprise": 0.1,
        },
    ]

    response = test_client.get("/sentiment/month")
    assert response.status_code == 200
    assert response.json() == fake_repo.get_recent_sentiment_history.return_value
    fake_repo.get_recent_sentiment_history.assert_called_once_with(31)


class FakeRepoOK:
    def healthcheck(self):
        pass


class FakeRepoFail:
    def healthcheck(self):
        raise RuntimeError("!")


def test_warmup(client):
    test_client, _ = client
    from app.api.main import app, get_repo

    app.dependency_overrides[get_repo] = FakeRepoOK

    response = test_client.get("/_ah/warmup")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_warmup_failure(client):
    test_client, _ = client

    from app.api.main import app, get_repo

    app.dependency_overrides[get_repo] = FakeRepoFail

    resp = test_client.get("/_ah/warmup")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"
    assert "error" in resp.json()


@pytest.mark.parametrize("i", range(11))
def test_rate_limit(client, i):
    test_client, fake_repo = client
    fake_repo.get_latest_sentiment.return_value = {"joy": 0.8}

    resp = test_client.get("/sentiment/current")
    if i < 9:
        assert resp.status_code == 200
    else:
        assert resp.status_code == 429
