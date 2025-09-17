# file: app/llm_annotation/annotation_worker.py
import re
import gc
import logging
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Tuple

import torch
from google.cloud import firestore, storage
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from datasets import load_dataset

from app.ml.preprocessing import prepare_for_input
from app.config import AnnoWorkerSettings
from app.logging_setup import setup_logging

setup_logging()
settings = AnnoWorkerSettings()
log = logging.getLogger("annotation_worker")
metrics = Counter()


def load_pipeline(model_id: str, settings: AnnoWorkerSettings):
    tok = AutoTokenizer.from_pretrained(
        model_id,
        token=settings.HF_TOKEN,
        padding_side="left",
        model_max_length=settings.MAX_PROMPT_LEN,
    )

    if settings.LOAD_8BT:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=settings.HF_TOKEN,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=settings.HF_TOKEN,
            trust_remote_code=True,
            device_map="auto",
        )

    if getattr(model.config, "_attn_implementation", None) != "sdpa":
        model.config._attn_implementation = "sdpa"
    model.config.use_cache = False

    return pipeline("text-generation", model=model, tokenizer=tok)


def build_prompt(tok, title: str, body: str, comments: List[str]):
    rules = (
        "Score each sentiment as an integer 1..10 (1=absent, 5=noticeable, 10=dominant). "
        "Output ONLY JSON with keys: joy, sadness, anger, fear, love, surprise."
    )

    content = f"""You are an expert annotator of emotional tone in Reddit content.
            {rules}
            {prepare_for_input(title, body, comments)}
            Return JSON only."""

    messages = [{"role": "user", "content": content}]

    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def torch_bootstrap(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("medium")


def generate_with_adaptive_bs(
    pipe, prompts: list[str], base_bs: int, max_new_tokens: int
):
    bs = base_bs
    i = 0
    outputs = []
    while i < len(prompts):
        take = min(bs, len(prompts) - i)
        batch = prompts[i : i + take]
        try:
            with torch.inference_mode():
                out = pipe(
                    batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_full_text=False,
                )
            outputs.extend(out)
            i += take
            if bs < base_bs:
                bs = min(base_bs, bs * 2)
        except torch.cuda.OutOfMemoryError:
            log.warning(f"CUDA OOM; reducing batch_size: {bs} -> {max(1, bs // 2)}")
            metrics["oom_events"] += 1
            del batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            bs = max(1, bs // 2)
            if bs == 1:
                log.error("Batch size is 1 and still OOM; cannot proceed")
                raise
    return outputs


def annotate_batch(
    dataset: Dict[str, List[Any]], pipeline, batch_size
) -> List[Tuple[str, str]]:
    """Annotate a batch of Reddit post + comments.

    Args:
        dataset (Dict[str, List[Any]]): A dict of columns from huggingface dataset
        pipeline (_type_): The text generation pipeline to use for annotation.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the post ID and the annotated JSON string.
        examples: [
            ('1mpauih','{"joy": 1, "sadness": 5, "anger": 10, "fear": 3, "love": 1, "surprise": 2}'),
            ('1mp2r91','{"joy": 1, "sadness": 5, "anger": 10, "fear": 5, "love": 1, "surprise": 2}')
            ]
    """
    prompts = []
    ids = dataset["id"]
    for title, text, comments in zip(
        dataset["title"], dataset["text"], dataset["comments"]
    ):
        comments = [comment["body"] for comment in comments]
        prompts.append(build_prompt(title, text, comments))
    outputs = generate_with_adaptive_bs(
        pipeline, prompts, base_bs=batch_size, max_new_tokens=settings.MAX_NEW_TOKENS
    )
    # [[{'generated_text': '{"joy": 1, "sadness": 2, "anger": 5, "fear": 3, "love": 1, "surprise": 6}'}],
    # [{'generated_text': '{"joy": 1, "sadness": 5, "anger": 10, "fear": 7, "love": 1, "surprise": 3}'}]]
    outputs = [parse_json(o[0]["generated_text"]) for o in outputs]
    metrics["json_parse_failures"] += sum(1 for o in outputs if o is None)
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
                            "lease_owner": settings.WORKER_ID,
                            "lease_expires_at": now2
                            + timedelta(minutes=settings.LEASE_MIN),
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
            "lease_owner": settings.WORKER_ID,
            "lease_expires_at": now + timedelta(minutes=settings.LEASE_MIN),
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
    return f"{settings.GCS_PREFIX}/{settings.RUN_ID}/{task_id}/{settings.WORKER_ID}"


def main():
    torch_bootstrap()
    db = firestore.Client(database=settings.FIRESTORE_DATABASE_ID)
    run_ref = db.collection(settings.FIRESTORE_ANNO_COLLECTIONS).document(
        settings.RUN_ID
    )
    tasks_ref = run_ref.collection(settings.FIRESTORE_TASKS_SUBCOLLECTIONS)
    run_config = run_ref.get().to_dict()
    # load dataset
    ds = load_dataset(
        settings.SOURCE_HF_REPO, split="train", revision=run_config.get("revision")
    )
    pipe = load_pipeline(settings.ANN_MODEL_ID, settings)
    gcs = storage.Client()
    bucket = gcs.bucket(settings.GCS_BUCKET)

    while True:
        log.info("[worker] Initializing Firestore client")
        log.info("[worker] Firestore client initialized")
        shard_id = lease_one(db, tasks_ref)
        if not shard_id:
            log.critical("[worker] no tasks available, exiting")
            break
        log.info(f"[worker] got shard {shard_id}")
        shard = tasks_ref.document(shard_id).get().to_dict()
        start_idx = shard["start_idx"]
        chunks_done = shard.get("chunk_done", 0)
        end_idx = shard["end_idx"]

        # process in chunks
        for idx in range(start_idx + chunks_done, end_idx + 1, settings.CHUNK_SIZE):
            hi = min(idx + settings.CHUNK_SIZE - 1, end_idx)
            log.info(f"[worker] processing chunk {idx} to {hi}")
            chunk = ds[idx : hi + 1]
            log.info(f"[worker] chunk size: {len(chunk)}")
            out = annotate_batch(chunk, pipe, settings.BATCH_SIZE)
            log.info(f"[worker] annotated {len(out)} items")
            records = []
            for pid, scores in out:
                records.append(
                    {
                        "id": pid,
                        "scores": scores,
                        "model_id": settings.ANN_MODEL_ID,
                        "run_id": settings.RUN_ID,
                        "worker_id": settings.WORKER_ID,
                    }
                )
            metrics["items_annotated"] += len(records)
            payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)

            blob_name = f"{_gcs_prefix(shard_id)}/chunk-{idx:07d}-{hi:07d}.jsonl"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(
                payload, content_type="application/jsonl", if_generation_match=0
            )
            log.info(f"[worker] uploaded chunk to {blob_name}")
            heartbeat(tasks_ref, shard_id, inc_done=len(records))
            del chunk, out, records, payload
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        mark_completed(tasks_ref, shard_id)
        log.info(f"[worker] shard {shard_id} done")
        log.info(f"[worker] metrics: {dict(metrics)}")


if __name__ == "__main__":
    main()
