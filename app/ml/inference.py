# File: app/ml/inference.py
import os

from transformers import pipeline

BERT_MAX_TOKEN = 512
classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    top_k=None,
)


def run_batch_inference(texts: list[str], batch_size: int = 32) -> list[dict]:
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if os.getenv("APP_ENV") == "test":
            print(f"batch: {batch}")
        truncated = [text[:BERT_MAX_TOKEN] for text in batch]
        results = classifier(truncated)

        all_results.extend([{res["label"]: res["score"] for res in r} for r in results])

    return all_results
