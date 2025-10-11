from types import SimpleNamespace


from app.ml import inference


def setup_module(module):
    inference.get_classifier.cache_clear()


def teardown_module(module):
    inference.get_classifier.cache_clear()


def test_get_classifier_uses_cache(monkeypatch):
    calls = []

    def fake_pipeline(task, model, truncation, top_k=None):
        calls.append((task, model, truncation, top_k))
        return f"pipeline-{len(calls)}"

    monkeypatch.setattr(inference, "pipeline", fake_pipeline)
    monkeypatch.setattr(
        inference,
        "settings",
        SimpleNamespace(SENTIMENT_MODEL_ID="model-A", BATCH_MAX_TOKENS=128),
    )

    first = inference.get_classifier()
    second = inference.get_classifier()

    assert first == second == "pipeline-1"
    assert calls == [("text-classification", "model-A", True, None)]


def test_run_batch_inference_truncates_and_flattens(monkeypatch):
    captured = []

    class DummyPipeline:
        def __call__(self, texts):
            captured.append(list(texts))
            return [
                [
                    {"label": "joy", "score": 0.9},
                    {"label": "sadness", "score": 0.1},
                ]
                for _ in texts
            ]

    monkeypatch.setattr(inference, "get_classifier", lambda: DummyPipeline())
    monkeypatch.setattr(
        inference,
        "settings",
        SimpleNamespace(BATCH_MAX_TOKENS=5, SENTIMENT_MODEL_ID="model-A"),
    )

    texts = ["abcdefgh", "ijklmnop"]
    results = inference.run_batch_inference(texts, batch_size=1)

    assert captured == [["abcde"], ["ijklm"]]
    assert results == [
        {"joy": 0.9, "sadness": 0.1},
        {"joy": 0.9, "sadness": 0.1},
    ]
