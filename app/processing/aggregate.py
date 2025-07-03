# File: app/processing/aggregate.py
from collections import defaultdict
import heapq

import numpy as np


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
    temperature = 1000  # Try tuning between 100–5000

    post_scores = np.array([max(post["score"], 0) for post in valid_posts])
    max_post_score = max(post_scores)

    normalized_post_scores = post_scores / temperature
    normalized_post_scores = post_scores - max_post_score

    weights = np.exp(normalized_post_scores)
    weights /= np.sum(weights)
    print(f"weights: {weights}")
    weighted_totals = defaultdict(float)
    top_contributors = defaultdict(list)

    for idx, post in enumerate(valid_posts):
        sentiment = post.get("sentiment")
        weight = weights[idx]
        for k, v in sentiment.items():
            contribution = v * weight
            weighted_totals[k] += contribution

            heapq.heappush(top_contributors[k], (contribution, idx, post))

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
