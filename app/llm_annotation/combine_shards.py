"""A script for combining shards"""

import argparse

from google.cloud import firestore

from utils.utils import getenv_str


def main():
    FIRESTORE_DATABASE_ID = getenv_str("FIRESTORE_DATABASE_ID", None)
    FIRESTORE_ANNO_COLLECTION_NAME = getenv_str(
        "FIRESTORE_ANNO_COLLECTION_NAME", "annotation_runs"
    )
    FIRESTORE_ANNO_TASKS_SUBCOLLECTION_NAME = getenv_str(
        "FIRESTORE_ANNO_TASKS_SUBCOLLECTION_NAME", "tasks"
    )
    # ANNO_BUCKET_NAME = getenv_str("ANNO_BUCKET_NAME", None)
    # ANN_BUCKET_PREFIX = getenv_str("ANN_BUCKET_PREFIX", None)
    # TARGET_REPO_ID = getenv_str("TARGET_REPO_ID", "Nech-C/reddit-sentiment-annotated")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        help="Annotation run name",
        required=True,
    )

    args = parser.parse_args()

    RUN_NAME = args.run_name
    # get shard and run info from firestore
    db = firestore.Client(database=FIRESTORE_DATABASE_ID)
    run_config = (
        db.collection(FIRESTORE_ANNO_COLLECTION_NAME).document(RUN_NAME).get().to_dict()
    )
    shard_size = run_config.get("shard_size", None)

    shards_info = (
        db.collection(FIRESTORE_ANNO_COLLECTION_NAME)
        .document(RUN_NAME)
        .collection(FIRESTORE_ANNO_TASKS_SUBCOLLECTION_NAME)
    )
    shards = [shard.to_dict() for shard in shards_info.get()]

    print(f"Found {len(shards)} shards in run {RUN_NAME}, shard_size={shard_size}")
