from collections import defaultdict


def compute_sentiment_average(posts: list[dict]) -> dict:
    """
    Aggregates sentiment scores across all posts.

    Args:
        posts (list): List of posts with 'sentiment' field (dict of 6 emotions).

    Returns:
        dict: Averaged sentiment values.
    """
    if not posts:
        return {}

    totals = defaultdict(float)
    count = 0

    for post in posts:
        sentiment = post.get("sentiment")
        if not sentiment:
            continue
        for k, v in sentiment.items():
            totals[k] += v
        count += 1

    return {k: v / count for k, v in totals.items()} if count else {}
