# File: app/processing/aggregate.py
from collections import defaultdict
import heapq

import numpy as np


def normalized_softmax(x: np.ndarray, temperature: int) -> np.ndarray:
    x = np.log1p(x)
    x /= temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def compute_sentiment_average(posts: list[dict]) -> dict:
    """
    Aggregates sentiment scores across all posts.

    Args:
        posts (list): List of posts with 'sentiment' field (dict of 6 emotions).

    Returns:
        dict: Averaged sentiment values.
    """
    valid_posts = [p for p in posts if "sentiment" in p and "score" in p]
    if not valid_posts:
        return {}

    # softmax
    temperature = 1.5  # Try tuning between 100–5000

    filtered = [(i, max(p["score"], 0)) for i, p in enumerate(valid_posts) if p["score"] > 0]
    indices, scores = zip(*filtered) if filtered else ([], [])
    scores = np.array(scores)
    weights = normalized_softmax(scores, temperature)
    
    weighted_totals = defaultdict(float)
    top_contributors = defaultdict(list)

    for i, w in zip(indices, weights):
        post = valid_posts[i]
        sentiment = post.get("sentiment")
        for k, v in sentiment.items():
            contribution = v * w
            weighted_totals[k] += contribution

            heapq.heappush(top_contributors[k], (contribution, i, post))

            if len(top_contributors[k]) > 3:
                heapq.heappop(top_contributors[k])

    total = sum(weighted_totals.values())
    if total == 0:
        return {k: 0 for k in weighted_totals}

    average = {k: v / total for k, v in weighted_totals.items()}

    return {
        **average,
        "_top_contributor": {
            k: [entry[2] | {"contribution": entry[0]} for entry in v]
            for k, v in top_contributors.items()
        },
    }
