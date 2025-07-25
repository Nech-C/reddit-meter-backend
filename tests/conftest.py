import pytest


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
