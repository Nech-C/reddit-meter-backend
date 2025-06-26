from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    top_k=None,
)


def run_batch_inference(texts: list[str]) -> list[dict]:
    # Truncate each to 512 characters
    truncated_texts = [text[:512] for text in texts]
    results = classifier(truncated_texts)

    return [{res["label"]: res["score"] for res in result} for result in results]
