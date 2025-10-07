# tests/test_models_post.py
import math
from datetime import datetime
import pytest
from pydantic import ValidationError

import app.models.post as post


def test_PostComment_success():
    """Ensure PostComment can be created with minimal and full data."""
    # minimum
    comment = post.PostComment.model_validate({"body": "I like pizza!"})
    assert comment is not None
    assert comment.body == "I like pizza!"
    assert comment.author is None
    assert comment.score is None

    # all fields
    comment2 = post.PostComment.model_validate(
        {"body": "hello!", "author": "me", "score": 100, "created_utc": 123345}
    )
    assert comment2.body == "hello!"
    assert comment2.author == "me"
    assert comment2.score == 100
    assert comment2.created_utc == 123345


def test_postcomment_failure():
    """Ensure PostComment rejects invalid input."""
    # missing required `body` fails
    with pytest.raises(ValidationError):
        post.PostComment.model_validate({})

    # score < 0 fails (model uses ge=0 on score)
    with pytest.raises(ValidationError):
        post.PostComment.model_validate({"body": "hi", "score": -1})


def test_sentiment_success_and_normalization():
    """Balanced sentiment (sum==1) and normalization behavior."""
    ONE_SIXTH = 1.0 / 6
    bal_sentiment_dict = {
        "joy": ONE_SIXTH,
        "sadness": ONE_SIXTH,
        "anger": ONE_SIXTH,
        "love": ONE_SIXTH,
        "surprise": ONE_SIXTH,
        "fear": ONE_SIXTH,
    }
    s = post.Sentiment.model_validate(bal_sentiment_dict)
    assert s is not None
    # model_dump returns plain floats
    assert s.model_dump() == bal_sentiment_dict

    # unnormalized -> normalized so sum ~= 1
    unnorm = {
        "joy": 0.1,
        "sadness": 0.1,
        "anger": 0.2,
        "love": 0.2,
        "surprise": 0.4,
        "fear": 0.4,
    }
    s2 = post.Sentiment.model_validate(unnorm)
    vals = s2.model_dump()
    assert math.isclose(sum(vals.values()), 1.0, rel_tol=1e-9)
    # relationships from input preserved proportionally
    # joy == sadness, anger == 2 * joy, surprise == 4 * joy
    assert math.isclose(s2.joy, s2.sadness, rel_tol=1e-9)
    assert math.isclose(s2.anger, s2.joy * 2, rel_tol=1e-9)
    assert math.isclose(s2.surprise, s2.joy * 4, rel_tol=1e-9)


def test_sentiment_missing_value_handling():
    """If only some emotions provided, defaults fill and sum can be 1.0 or normalized."""
    # two values summing to 1 -> ok
    missing_sentiment = {"joy": 0.5, "sadness": 0.5}
    s = post.Sentiment.model_validate(missing_sentiment)
    assert s is not None
    assert math.isclose(sum(s.model_dump().values()), 1.0, rel_tol=1e-9)

    # if values sum to 0 (all omitted) it's allowed and remains zeros
    s_empty = post.Sentiment.model_validate({})
    total = sum(s_empty.model_dump().values())
    assert math.isclose(total, 0.0, rel_tol=1e-9)


def test_sentiment_failure_on_invalid_probabilities():
    invalid = {
        "joy": 2.0,
        "sadness": 0.1,
        "anger": 0.2,
        "love": 1.1,
        "surprise": 0.4,
        "fear": 0.4,
    }
    with pytest.raises(ValidationError):
        post.Sentiment.model_validate(invalid)


def test_post_timestamp_coercion_and_validation():
    """post_created_ts accepts epoch and becomes datetime; processing earlier fails."""
    epoch = 1_690_000_000  # some epoch seconds
    p = post.Post.model_validate({"post_id": "x", "post_created_ts": epoch})
    assert isinstance(p.post_created_ts, datetime)

    # explicitly set processing_timestamp earlier than post_created_ts -> validation error
    # use two different epoch values so processing < created
    created_epoch = 1_700_000_000
    processing_epoch_early = 1_600_000_000
    with pytest.raises(ValidationError):
        post.Post.model_validate(
            {
                "post_id": "x2",
                "post_created_ts": created_epoch,
                "processing_timestamp": processing_epoch_early,
            }
        )

    # if processing_timestamp omitted, field_validator returns a datetime (not None)
    p2 = post.Post.model_validate({"post_id": "p2", "post_created_ts": epoch})
    assert p2 is not None
    assert isinstance(p2.processing_timestamp, datetime)


def test_post_comments_default_is_fresh_per_instance():
    """Ensure default_factory gives each Post its own list instance."""
    a = post.Post.model_validate({"post_id": "a"})
    b = post.Post.model_validate({"post_id": "b"})
    assert a.post_comments == []
    assert b.post_comments == []
    a.post_comments.append(post.PostComment.model_validate({"body": "x"}))
    assert b.post_comments == []  # unchanged


def test_post_model_dump_json_compatible():
    """Ensure model_dump(mode='json') returns JSON-serializable structure for lists/records."""
    p = post.Post.model_validate(
        {
            "post_id": "abc",
            "post_title": "hello",
            "post_created_ts": 1_690_000_000,
            "post_comments": [
                {"body": "c1", "author": "u1", "score": 3, "created_utc": 123},
                {"body": "c2", "author": "u2"},
            ],
        }
    )
    dumped = p.model_dump(mode="json")
    # keys present
    assert dumped["post_id"] == "abc"
    assert isinstance(dumped["post_comments"], list)
    # nested comment is dict-like and contains fields
    first = dumped["post_comments"][0]
    assert first["body"] == "c1"
    # created_ts was converted to ISO string in JSON mode
    assert isinstance(dumped.get("post_created_ts"), str)
