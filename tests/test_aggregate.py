import pytest
import numpy as np

from app.processing.aggregate import normalized_softmax, compute_sentiment_average


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


def test_normalized_softmax():
    # simulate a power law distribution
    count = 100
    scores = []
    for n in range(6):
        scores.extend([n * 100] * int(count))
        count /= 2

    scores = np.array(scores)
    out = normalized_softmax(scores, 2)

    assert np.isclose(np.sum(out), 1.0)
    assert all(out > 0.0)


def test_compute_sentiment_average_all_emotions(sample_posts):
    result = compute_sentiment_average(sample_posts)

    emotions = ["joy", "sadness", "anger", "fear", "love", "surprise"]
    # All emotions present
    assert set(emotions).issubset(result.keys()), "Missing one of the six emotions"
    assert "_top_contributor" in result, "Missing '_top_contributor' in result"

    # Sum to ~1
    total = sum(result[e] for e in emotions)
    assert pytest.approx(total, rel=1e-6) == 1.0, f"Expected total≈1.0, got {total}"

    # Test that for each emotion, top_contributors length <= 3 and entries valid
    tc = result["_top_contributor"]
    assert set(emotions) == set(tc.keys()), (
        "Every emotion must have a top-contributor list"
    )
    for e in emotions:
        contribs = tc[e]
        assert 1 <= len(contribs) <= 3, (
            f"Top contributors for {e} should be 1–3, got {len(contribs)}"
        )
        for entry in contribs:
            # Each entry must merge the original post dict and add 'contribution'
            assert "contribution" in entry and isinstance(entry["contribution"], float)
            assert entry["id"] in {f"p{i}" for i in range(1, 6)}, (
                "Contributor id must be one of p1–p5"
            )


def test_compute_sentiment_average_zero_total():
    # All sentiments zero but positive score => triggers total==0 branch
    posts = [
        {
            "sentiment": {
                e: 0.0 for e in ["joy", "sadness", "anger", "fear", "love", "surprise"]
            },
            "score": 5,
        }
    ]
    result = compute_sentiment_average(posts)
    # Each emotion should be 0 and no _top_contributor key
    for e in ["joy", "sadness", "anger", "fear", "love", "surprise"]:
        assert e in result and result[e] == 0
    assert "_top_contributor" not in result, (
        "_top_contributor should be absent when total==0"
    )


@pytest.mark.parametrize(
    "posts",
    [
        [],  # no posts
        [
            {"score": 5},
            {
                "sentiment": {
                    "joy": 1,
                    "sadness": 0,
                    "anger": 0,
                    "fear": 0,
                    "love": 0,
                    "surprise": 0,
                }
            },
        ],  # all invalid
    ],
)
def test_compute_sentiment_average_no_valid(posts):
    assert compute_sentiment_average(posts) == {}
