# file: app/llm_annotation/annotation_worker.py
import os
import re
import json
import gc
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Tuple

import torch
from dotenv import load_dotenv
from google.cloud import firestore, storage
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from datasets import load_dataset

from app.utils.utils import getenv_int, getenv_str, getenv_bool, get_dotenv_name
from app.ml.preprocessing import prepare_for_input

APP_ENV = get_dotenv_name()
load_dotenv(get_dotenv_name())

RUN_ID = getenv_str("RUN_ID")
WORKER_ID = getenv_str("WORKER_ID", f"worker-{os.getpid()}")
HF_TOKEN = getenv_str("HF_TOKEN")
ANN_MODEL_ID = getenv_str("ANN_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
SOURCE_HF_REPO = getenv_str("SOURCE_REPO_ID", "Nech-C/reddit-sentiment")
GCS_BUCKET = getenv_str("ANNO_BUCKET")
GCS_PREFIX = getenv_str("ANNO_BUCKET_PREFIX", "annotations")
LEASE_MIN = getenv_int("LEASE_MIN", 30)
LOAD_8BT = getenv_bool("LOAD_8BT", True)
FIRESTORE_DATABASE_ID = getenv_str("FIRESTORE_DATABASE_ID", "sentiment-db")
FIRESTORE_ANNO_COLLECTIONS = getenv_str(
    "FIRESTORE_ANNO_COLLECTION_NAME", "annotation_runs"
)
FIRESTORE_TASKS_SUBCOLLECTIONS = getenv_str(
    "FIRESTORE_ANNO_TASKS_SUBCOLLECTION_NAME", "tasks"
)
MAX_PROMPT_LEN = getenv_int("MAX_PROMPT_LEN", 1024)
MAX_NEW_TOKENS = getenv_int("MAX_NEW_TOKENS", 16)
CHUNK_SIZE = getenv_int("CHUNK_SIZE", 1024)
BATCH_SIZE = getenv_int("BATCH_SIZE", 8)
assert RUN_ID and GCS_BUCKET and HF_TOKEN, (
    "RUN_ID, GCS_BUCKET, and HF_TOKEN must be set"
)


def load_pipeline(model_id: str):
    tok = AutoTokenizer.from_pretrained(
        ANN_MODEL_ID,
        use_auth_token=HF_TOKEN,
        padding_side="left",
        model_max_length=MAX_PROMPT_LEN,
    )
    if LOAD_8BT:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            ANN_MODEL_ID,
            use_auth_token=HF_TOKEN,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ANN_MODEL_ID,
            use_auth_token=HF_TOKEN,
            trust_remote_code=True,
            device_map="auto",
        )

    return pipeline("text-generation", model=model, tokenizer=tok)


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
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE | re.IGNORECASE).strip()
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


def safe_call(pipe, inputs, **kwargs):
    try:
        with torch.inference_mode():  # stricter than no_grad, no autograd state
            return pipe(inputs, **kwargs)  # e.g., text-generation pipeline call
    except torch.cuda.OutOfMemoryError:
        # Clear as much as possible, then re-raise so caller can back off (smaller batch/length)
        del inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # helps with inter-process handles
        raise
    finally:
        # Always drop references from this scope (even on success)
        try:
            del inputs
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def annotate_batch(
    cols: Dict[str, List[Any]], pipeline, batch_size
) -> List[Tuple[str, str]]:
    """Annotate a batch of Reddit post + comments.

    Args:
        cols (Dict[str, List[Any]]): A dict of columns from huggingface dataset
        pipeline (_type_): The text generation pipeline to use for annotation.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the post ID and the annotated JSON string.
        examples: [
            ('1mpauih','{"joy": 1, "sadness": 5, "anger": 10, "fear": 3, "love": 1, "surprise": 2}'),
            ('1mp2r91','{"joy": 1, "sadness": 5, "anger": 10, "fear": 5, "love": 1, "surprise": 2}')
            ]
    """
    prompts = []
    ids = cols["id"]
    for title, text, comments in zip(cols["title"], cols["text"], cols["comments"]):
        comments = [comment["body"] for comment in comments]
        prompts.append([build_prompt(title, text, comments)])
    with torch.inference_mode():
        outputs = []
        for idx in range(0, len(prompts), batch_size):
            output = safe_call(
                pipeline,
                prompts[idx : idx + batch_size],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                padding="max_length",
                return_full_text=False,
                batch_size=batch_size,
            )
            outputs.extend(output)
        # [[{'generated_text': '{"joy": 1, "sadness": 2, "anger": 5, "fear": 3, "love": 1, "surprise": 6}'}],
        # [{'generated_text': '{"joy": 1, "sadness": 5, "anger": 10, "fear": 7, "love": 1, "surprise": 3}'}]]
        outputs = [parse_json(output[0]["generated_text"]) for output in outputs]

    out = list(zip(ids, outputs))
    return out


def lease_one(db: firestore.Client, tasks_ref) -> Optional[str]:
    now = datetime.now(timezone.utc)
    # Pull a small window to reduce conflicts
    candidates = list(
        tasks_ref.where("status", "in", ["PENDING", "IN_PROGRESS"])
        .order_by("updated_at")
        .limit(25)
        .stream()
    )
    for snap in candidates:
        d = snap.to_dict()
        expired = (d.get("lease_expires_at") is None) or (d["lease_expires_at"] < now)
        if d["status"] == "PENDING" or expired:

            @firestore.transactional
            def txn(tx):
                s2 = snap.reference.get(transaction=tx)
                d2 = s2.to_dict()
                now2 = datetime.now(timezone.utc)
                expired2 = (d2.get("lease_expires_at") is None) or (
                    d2["lease_expires_at"] < now2
                )
                if d2["status"] == "PENDING" or expired2:
                    tx.update(
                        snap.reference,
                        {
                            "status": "IN_PROGRESS",
                            "lease_owner": WORKER_ID,
                            "lease_expires_at": now2 + timedelta(minutes=LEASE_MIN),
                            "updated_at": now2,
                            "attempts": d2.get("attempts", 0)
                            + (1 if d2["status"] == "PENDING" else 0),
                        },
                    )
                    return snap.reference.id
                return None

            res = txn(db.transaction())
            if res:
                return res
    return None


def heartbeat(tasks_ref, doc_id: str, inc_done: int = 0):
    now = datetime.now(timezone.utc)
    tasks_ref.document(doc_id).update(
        {
            "lease_owner": WORKER_ID,
            "lease_expires_at": now + timedelta(minutes=LEASE_MIN),
            "updated_at": now,
            "chunk_done": firestore.Increment(inc_done),
        }
    )


def mark_completed(tasks_ref, doc_id: str):
    tasks_ref.document(doc_id).update(
        {
            "status": "COMPLETED",
            "lease_owner": None,
            "lease_expires_at": None,
            "updated_at": datetime.now(timezone.utc),
        }
    )


def _gcs_prefix(task_id):
    return f"{GCS_PREFIX}/{RUN_ID}/{task_id}/{WORKER_ID}"


def main():
    while True:
        load_dotenv(APP_ENV)
        print("[worker] Initializing Firestore client")
        db = firestore.Client(database=FIRESTORE_DATABASE_ID)
        print("[worker] Firestore client initialized")
        run_ref = db.collection(FIRESTORE_ANNO_COLLECTIONS).document(RUN_ID)
        tasks_ref = run_ref.collection(FIRESTORE_TASKS_SUBCOLLECTIONS)
        run_config = run_ref.get().to_dict()

        shard_id = lease_one(db, tasks_ref)
        if not shard_id:
            print("[worker] no tasks available, exiting")
            break
        print(f"[worker] got shard {shard_id}")
        shard = tasks_ref.document(shard_id).get().to_dict()
        start_idx = shard["start_idx"]
        end_idx = shard["end_idx"]

        # load dataset
        ds = load_dataset(
            SOURCE_HF_REPO, split="train", revision=run_config.get("revision")
        )
        pipe = load_pipeline(ANN_MODEL_ID)

        gcs = storage.Client()
        bucket = gcs.bucket(GCS_BUCKET)
        # process in chunks
        for idx in range(start_idx, end_idx + 1, CHUNK_SIZE):
            hi = min(idx + CHUNK_SIZE - 1, end_idx)
            print(f"[worker] processing chunk {idx} to {hi}")
            chunk = ds[idx : hi + 1]
            print(f"[worker] chunk size: {len(chunk)}")
            out = annotate_batch(chunk, pipe, BATCH_SIZE)
            print(f"[worker] annotated {len(out)} items")
            records = []
            for pid, scores in out:
                records.append(
                    {
                        "id": pid,
                        "scores": scores,
                        "model_id": ANN_MODEL_ID,
                        "run_id": RUN_ID,
                        "worker_id": WORKER_ID,
                    }
                )
            payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)

            blob_name = f"{_gcs_prefix(shard_id)}/chunk-{idx:07d}-{hi:07d}.jsonl"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(payload, content_type="application/jsonl")
            print(f"[worker] uploaded chunk to {blob_name}")
            heartbeat(tasks_ref, shard_id, inc_done=(hi - idx + 1))

        mark_completed(tasks_ref, shard_id)
        print(f"[worker] shard {shard_id} done")


if __name__ == "__main__":
    main()
