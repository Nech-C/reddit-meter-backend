# file: app/llm_annotation/annotation_worker.py
import os
import re
import json
from typing import List, Dict, Optional, Any

from utils.utils import getenv_int, getenv_str, getenv_bool
from google.cloud import firestore, storage
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from app.ml.preprocessing import prepare_for_input

RUN_ID = getenv_str("RUN_ID")
WORKER_ID = getenv_str("WORKER_ID", f"worker-{os.getpid()}")
HF_TOKEN = getenv_str("HF_TOKEN")
MODEL_ID = getenv_str("ANN_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
SOURCE_HF_REPO = getenv_str("SOURCE_REPO_ID", "Nech-C/reddit-sentiment")
GCS_BUCKET = getenv_str("ANN_BUCKET")
GCS_PREFIX = getenv_str("ANN_BUCKET_PREFIX", "annotations")
LEASE_MIN = getenv_int("LEASE_MIN", "30")
LOAD_4BT = getenv_bool("LOAD_4BT", True)
MAX_NEW_TOK = None

assert RUN_ID and GCS_BUCKET and HF_TOKEN, (
    "RUN_ID, GCS_BUCKET, and HF_TOKEN must be set"
)


db = firestore.client()
run_ref = db.collection("annotation_runs").document(RUN_ID)
tasks_ref = run_ref.collection("tasks")

gcs = storage.Client()
bucket = gcs.bucket(GCS_BUCKET)


def load_llm(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    if LOAD_4BT:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            use_auth_token=HF_TOKEN,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, use_auth_token=HF_TOKEN, trust_remote_code=True, device_map="auto"
        )

    return tokenizer, model


def build_prompt(title: str, body: str, comments: List[str]):
    rules = (
        "Score each sentiment as an integer 1..10 (1=absent, 5=noticeable, 10=dominant). "
        "Output ONLY JSON with keys: joy, sadness, anger, fear, love, surprise."
    )

    prompt_content = f"""You are an expert annotator of emotional tone in Reddit content.
            {rules}
            {prepare_for_input(title, body, comments)}
            Return JSON only."""

    prompt = {"role": "user", "content": prompt_content}

    return prompt


def parse_json(text: str) -> Optional[Dict[str, int]]:
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        out = {}
        for k in ["joy", "sadness", "anger", "fear", "love", "surprise"]:
            v = int(obj.get(k, 1))
            out[k] = max(1, min(10, v))
        return out
    except Exception:
        return None


def annotate_batch(rows: List[Dict[str, Any]], tok, model) -> List[Dict[str, Any]]:
    prompts = []
    ids = []
    for r in rows:
        title = r.get("title") or ""
        body = r.get("selftext") or r.get("text") or ""
        comments_raw = r.get("comments") or []
        comments = []
        for c in comments_raw[:5]:
            if isinstance(c, str):
                comments.append(c)
            elif isinstance(c, dict) and "body" in c:
                comments.append(c["body"])
        prompts.append(build_prompt(title, body, comments))
        ids.append(r.get("id"))

    inputs = tok(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOK,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tok.eos_token_id,
        )
    texts = tok.batch_decode(outputs, skip_special_tokens=True)
    out = []
    for rid, t in zip(ids, texts):
        obj = parse_json(t)
        out.append({"post_id": rid, **(obj or {})})
    return out
