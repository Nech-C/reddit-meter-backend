from collections import Counter
from types import SimpleNamespace
import json

import pytest

from app.llm_annotation import annotation_worker as aw


@pytest.fixture(autouse=True)
def reset_metrics():
    aw.metrics = Counter()
    yield
    aw.metrics = Counter()


def test_build_prompt_includes_cleaned_text(monkeypatch):
    fake_tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **_: "rendered"
    )
    monkeypatch.setattr(
        aw, "prepare_for_input", lambda title, body, comments: "CLEANED"
    )

    prompt = aw.build_prompt(fake_tokenizer, "title", "body", ["c1", "c2"])
    assert prompt == "rendered"


def test_parse_json_variants():
    standard = '{"joy": 1, "sadness": 2, "anger": 7, "fear": 3, "love": 1, "surprise": 5}'
    assert aw.parse_json(standard) == {
        "joy": 1,
        "sadness": 2,
        "anger": 7,
        "fear": 3,
        "love": 1,
        "surprise": 5,
    }

    wrapped = "```json{\"joy\": 1, \"sadness\": 2, \"anger\": 7, \"fear\": 3, \"love\": 1, \"surprise\": 5}```"
    assert aw.parse_json(wrapped) == {
        "joy": 1,
        "sadness": 2,
        "anger": 7,
        "fear": 3,
        "love": 1,
        "surprise": 5,
    }

    out_of_bounds = '{"joy": -1, "sadness": 12, "anger": 0, "fear": 11, "love": 1, "surprise": 25}'
    assert aw.parse_json(out_of_bounds) == {
        "joy": 1,
        "sadness": 10,
        "anger": 1,
        "fear": 10,
        "love": 1,
        "surprise": 10,
    }

    assert aw.parse_json("not json") is None


def test_annotate_batch_builds_prompts_and_parses(monkeypatch):
    dataset = {
        "id": ["abc123"],
        "title": ["My title"],
        "text": ["Body text"],
        "comments": [[{"body": "Comment 1"}, {"body": "Comment 2"}]],
    }

    captured = {}

    def fake_build_prompt(title, body, comments):
        captured["args"] = (title, body, comments)
        return "prompt"

    def fake_generate(pipe, prompts, base_bs, max_new_tokens):
        assert prompts == ["prompt"]
        captured["batch"] = (pipe, base_bs, max_new_tokens)
        payload = json.dumps(
            {
                "joy": 5,
                "sadness": 4,
                "anger": 3,
                "fear": 2,
                "love": 6,
                "surprise": 7,
            }
        )
        return [[{"generated_text": payload}]]

    monkeypatch.setattr(aw, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(aw, "generate_with_adaptive_bs", fake_generate)

    settings = SimpleNamespace(MAX_NEW_TOKENS=42)
    pipe = object()

    result = aw.annotate_batch(settings, dataset, pipe, batch_size=3)

    assert captured["args"] == (
        "My title",
        "Body text",
        ["Comment 1", "Comment 2"],
    )
    assert captured["batch"] == (pipe, 3, 42)
    assert result == [
        (
            "abc123",
            {
                "joy": 5,
                "sadness": 4,
                "anger": 3,
                "fear": 2,
                "love": 6,
                "surprise": 7,
            },
        )
    ]
    assert aw.metrics["json_parse_failures"] == 0


def test_gcs_prefix_builds_expected_path():
    settings = SimpleNamespace(
        GCS_PREFIX="annotations",
        RUN_ID="run1",
        WORKER_ID="workerA",
    )
    prefix = aw._gcs_prefix(settings, "shard42")
    assert prefix == "annotations/run1/shard42/workerA"
