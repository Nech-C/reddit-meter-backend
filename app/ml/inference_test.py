# File: app/ml/inference.py
from transformers import pipeline
from memory_profiler import profile

classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    top_k=None,
    device="cpu",
    batch_size=20000,
)


@profile
def run_batch_inference(texts: list[str], batch_size) -> list[dict]:
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        truncated = [text[:512] for text in batch]
        results = classifier(truncated)

        all_results.extend([{res["label"]: res["score"] for res in r} for r in results])

    return all_results


if __name__ == "__main__":
    texts = ["hi" * 1000] * 20000
    results = run_batch_inference(texts, batch_size=20000)
