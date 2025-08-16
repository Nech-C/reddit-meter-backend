# File: app/reddit/create_shards.py
import math
from datetime import datetime, timezone

from dotenv import load_dotenv
from datasets import load_dataset
from google.cloud import firestore

from app.utils.utils import getenv_int, getenv_str, get_dotenv_name


def main():
    load_dotenv(get_dotenv_name())

    SOURCE_REPO_ID = getenv_str("SOURCE_REPO_ID", "Nech-C/reddit-sentiment")
    TARGET_REPO_ID = getenv_str("TARGET_REPO_ID", "Nech-C/reddit-sentiment-annotated")
    DS_SPLIT = "train"
    SHARD_SIZE = getenv_int("SHARD_SIZE", 2048)
    CHUNK_SIZE = getenv_int("CHUNK_SIZE", 256)
    ANN_MODEL_ID = getenv_str("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    FIRESTORE_DATABASE_ID = getenv_str("FIRESTORE_DATABASE_ID", None)
    RUN_ID = input("Enter RUN_ID:")
    REVISION = input("Enter REVISION:")

    ds = load_dataset(SOURCE_REPO_ID, revision=REVISION, split=DS_SPLIT)
    n = len(ds)
    shards = math.ceil(n / SHARD_SIZE)

    print(
        f"Dataset len={n}, shards={shards} (size {SHARD_SIZE}), chunk_size={CHUNK_SIZE}"
    )

    db = firestore.Client(database=FIRESTORE_DATABASE_ID)
    run_doc = db.collection("annotation_runs").document(RUN_ID)

    run_doc.set(
        {
            "source_repo_id": SOURCE_REPO_ID,
            "revision": REVISION,
            "target_repo_id": TARGET_REPO_ID,
            "shard_size": SHARD_SIZE,
            "chunk_size": CHUNK_SIZE,
            "ann_model_id": ANN_MODEL_ID,
            "created_at": datetime.now(timezone.utc),
        },
        merge=True,
    )

    tasks = run_doc.collection("tasks")

    batch = db.batch()

    for i in range(shards):
        start = i * SHARD_SIZE
        end = min((i + 1) * SHARD_SIZE, n) - 1
        chunk_total = math.ceil((end - start + 1) / CHUNK_SIZE)

        doc_id = f"shard-{i + 1:06d}"
        docref = tasks.document(doc_id)
        batch.set(
            docref,
            {
                "status": "PENDING",
                "start_idx": start,
                "end_idx": end,
                "chunk_total": chunk_total,
                "chunk_done": 0,
                "lease_owner": None,
                "lease_expires_at": None,
                "attempts": 0,
                "updated_at": datetime.now(timezone.utc),
            },
            merge=False,
        )

        if i % 400 == 0:
            batch.commit()
            batch = db.batch()

    batch.commit()
    print("shards created.")


if __name__ == "__main__":
    main()
