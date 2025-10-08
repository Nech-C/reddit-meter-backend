# File: app/ml/inference.py
import os
from functools import lru_cache

from transformers import pipeline

from app.config import get_inference_settings

settings = get_inference_settings()


@lru_cache()
def get_classifier():
    """Get sentiment analysis pipeine with caching.

    Returns:
        Pipeline: a text-classification pipeline for sentiment analysis.
    """
    return pipeline(
        "text-classification",
        model=settings.SENTIMENT_MODEL_ID,
        truncation=True,
        top_k=None,
    )


def run_batch_inference(texts: list[str], batch_size: int = 32) -> list[dict]:
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if os.getenv("APP_ENV") == "test":
            print(f"batch: {batch}")
        truncated = [
            text[: settings.BATCH_MAX_TOKENS] for text in batch
        ]  # change BATCH_MAX_TOKEN
        results = get_classifier()(truncated)

        all_results.extend([{res["label"]: res["score"] for res in r} for r in results])

    return all_results
