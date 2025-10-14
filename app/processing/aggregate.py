# File: app/processing/aggregate.py
"""Aggregation helpers for computing sentiment summaries from Reddit posts."""

from collections import defaultdict
import heapq
from typing import Iterable

import numpy as np

from app.models.post import Post, SentimentSummary


def normalized_softmax(x: np.ndarray, temperature: int) -> np.ndarray:
    x = np.log1p(x)
    x /= temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _ensure_post(post_data: Post | dict) -> Post:
    """Coerce dictionaries into :class:`Post` models for uniform handling."""

    if isinstance(post_data, Post):
        return post_data
    return Post.model_validate(post_data)


def compute_sentiment_average(posts: Iterable[Post | dict]) -> SentimentSummary:
    """
    Aggregates sentiment scores across all posts.

    Args:
        posts (Iterable[Post | dict]): Reddit posts (models or dictionaries) with
            sentiment predictions attached.

    Returns:
        dict: Averaged sentiment values.
    """
    validated_posts = [_ensure_post(p) for p in posts]
    valid_posts = [
        p
        for p in validated_posts
        if p.sentiment is not None and p.post_score is not None
    ]
    if not valid_posts:
        return {}

    # softmax
    temperature = 1.5  # Try tuning between 100–5000

    filtered = [
        (i, max(p.post_score or 0, 0))
        for i, p in enumerate(valid_posts)
        if (p.post_score or 0) > 0
    ]
    if not filtered:
        return {}

    indices, scores = zip(*filtered)
    scores = np.array(scores)
    weights = normalized_softmax(scores, temperature)

    weighted_totals = defaultdict(float)
    top_contributors = defaultdict(list)

    for i, w in zip(indices, weights):
        post = valid_posts[i]
        sentiment = post.sentiment.model_dump(mode="python") if post.sentiment else {}
        for k, v in sentiment.items():
            contribution = v * w
            weighted_totals[k] += contribution

            heapq.heappush(top_contributors[k], (contribution, i, post))

            if len(top_contributors[k]) > 3:
                heapq.heappop(top_contributors[k])

    total = sum(weighted_totals.values())
    if total == 0:
        return {label: 0 for label in weighted_totals}

    averages = {label: val / total for label, val in weighted_totals.items()}
    return SentimentSummary.model_validate(
        {
            **averages,
            "top_contributors": [
                {
                    "emotion": emotion,
                    "top_posts": [
                        {
                            **post.to_python_dict(),
                            "contribution": contrib,
                        }
                        for contrib, _, post in sorted(entries, reverse=True)
                    ],
                }
                for emotion, entries in top_contributors.items()
            ],
        }
    )
