import pytest
import numpy as np

from app.processing.aggregate import normalized_softmax, compute_sentiment_average


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
