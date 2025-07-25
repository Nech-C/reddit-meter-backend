from fastapi.testclient import TestClient
from unittest.mock import patch

from app.api.main import app

app = TestClient(app)


def test_read_root():
    response = app.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}


@patch("app.api.main.get_latest_sentiment")
def test_current_sentiment(mock_get_latest):
    mock_get_latest.return_value = {
        "love": 0.02,
        "anger": 0.27,
        "sadness": 0.13,
        "joy": 0.43,
        "timestamp": "2025-07-25T01:40:48.024915+00:00",
        "surprise": 0.05,
    }

    response = app.get("/sentiment/current")
    assert response.status_code == 200
    assert response.json() == mock_get_latest.return_value


@patch("app.api.main.get_recent_sentiment_history")
def test_get_past_day_sentiment(mock_get_past_day):
    mock_get_past_day.return_value = [
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

    response = app.get("/sentiment/day")

    assert response.status_code == 200
    assert response.json() == [
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


@patch("app.api.main.get_recent_sentiment_history")
def test_get_past_week_sentiment(mock_get_week):
    mock_get_week.return_value = [
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

    response = app.get("/sentiment/week")
    assert response.status_code == 200
    assert response.json() == mock_get_week.return_value


@patch("app.api.main.get_recent_sentiment_history")
def test_get_past_month_sentiment(mock_get_month):
    mock_get_month.return_value = [
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

    response = app.get("/sentiment/month")
    assert response.status_code == 200
    assert response.json() == mock_get_month.return_value
