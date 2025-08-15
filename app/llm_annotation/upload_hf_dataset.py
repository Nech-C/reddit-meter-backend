#!/usr/bin/env python3
"""
Downloads Reddit post archives from GCS in parallel, builds a HF Dataset,
pushes it to the Hub, and (optionally) deletes the source blobs after success.

- Multithreaded downloads using ThreadPoolExecutor
- Tunable chunk size and worker count (env-based)
- Robust JSON loading and DataFrame prep
"""

import json
import os
import time
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage
from datasets import Dataset


# ---------------------------
# Env helpers
# ---------------------------
def getenv_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("true", "1", "yes", "on")


def getenv_int(key: str, default: int) -> int:
    v = os.getenv(key)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def getenv_str(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


# ---------------------------
# Multithreaded download
# ---------------------------
def _download_one(
    blob: storage.Blob,
    base_dir: Path,
    add_json_suffix: bool,
    chunk_size_bytes: int | None,
) -> Tuple[str, Path | None, int, Exception | None]:
    """
    Worker function: download one blob to base_dir.

    Returns (blob_name, local_path or None, bytes_written, error or None)
    """
    try:
        if chunk_size_bytes:
            # Larger chunks reduce per-request overhead for large objects.
            # Must be a multiple of 256 KB. 8 MB is a good general default.
            blob.chunk_size = chunk_size_bytes

        name = Path(blob.name)
        if add_json_suffix and name.suffix.lower() != ".json":
            name = name.with_name(name.name + ".json")

        local_path = base_dir / name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)
        return blob.name, local_path, local_path.stat().st_size, None
    except Exception as e:
        return blob.name, None, 0, e


def parallel_download(
    blobs: Iterable[storage.Blob],
    base_dir: Path,
    max_workers: int = 16,
    chunk_size_mb: int | None = 8,
    add_json_suffix: bool = True,
) -> Tuple[List[Path], int, List[Tuple[str, str]]]:
    """
    Download many blobs concurrently.

    Returns:
        files: list of local Paths
        total_bytes: sum of bytes written
        errors: list of (blob_name, error_repr)
    """
    chunk_size_bytes = int(chunk_size_mb * 1024 * 1024) if chunk_size_mb else None
    files: List[Path] = []
    total_bytes = 0
    errors: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_download_one, b, base_dir, add_json_suffix, chunk_size_bytes)
            for b in blobs
        ]
        for fut in as_completed(futures):
            name, path, size, err = fut.result()
            if err:
                errors.append((name, repr(err)))
            else:
                files.append(path)  # type: ignore[arg-type]
                total_bytes += size

    return files, total_bytes, errors


# ---------------------------
# JSON loading
# ---------------------------
def load_post_records(files: List[Path]) -> list:
    """
    Load JSON files. Accepts either a list of records or a single dict record.
    If your archives sometimes store under {"data": [...]}, we pull that out, too.
    """
    records: list = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list):
                    records.extend(data["data"])
                else:
                    records.append(data)
            else:
                print(f"Skipping {fp}: not list/dict JSON.")
        except Exception as e:
            print(f"Failed to parse {fp}: {e}. Skipped.")
    return records


# ---------------------------
# Main
# ---------------------------
def main() -> int:
    # Load env: base .env then overlay by APP_ENV
    load_dotenv()
    app_env = getenv_str("APP_ENV", "dev")
    if not load_dotenv(f".env.{app_env}"):
        print(f"Incorrect APP_ENV: {app_env}. Aborted.")
        return 1

    # Required envs
    bucket_name = getenv_str("GOOGLE_BUCKET_NAME")
    hf_token = getenv_str("HF_TOKEN")
    if not bucket_name:
        print("GOOGLE_BUCKET_NAME is not set.")
        return 1
    if not hf_token:
        print("HF_TOKEN is not set.")
        return 1

    # Behavior/env tunables
    min_archive_count = getenv_int("MIN_ARCHIVE_COUNT", 1)
    gcs_prefix = getenv_str("GCS_PREFIX", "")
    delete_after_upload = getenv_bool("DELETE_AFTER_UPLOAD", False)

    # Download tuning
    max_workers = getenv_int("DL_MAX_WORKERS", 16)
    chunk_size_mb = getenv_int("DL_CHUNK_MB", 8)
    tmp_dir = getenv_str("TMPDIR", "/tmp")

    storage_client = storage.Client()

    # List blobs (optionally by prefix)
    print(f"Listing blobs from bucket={bucket_name!r} prefix={gcs_prefix!r} …")
    blobs = list(storage_client.list_blobs(bucket_name, prefix=gcs_prefix))
    print(f"Found {len(blobs)} blobs.")
    if len(blobs) < min_archive_count:
        print(
            f"Required archives: {min_archive_count}; actual: {len(blobs)}. Aborting."
        )
        return 1

    with tempfile.TemporaryDirectory(dir=tmp_dir) as tempdirname:
        tempdir = Path(tempdirname)
        print(f"Temp dir: {tempdir}")

        # ---- Multithreaded download ----
        t0 = time.time()
        files, total_bytes, errors = parallel_download(
            blobs=blobs,
            base_dir=tempdir,
            max_workers=max_workers,
            chunk_size_mb=chunk_size_mb,
            add_json_suffix=True,
        )
        elapsed = time.time() - t0
        mib_s = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
        print(
            f"Downloaded {len(files)}/{len(blobs)} files, "
            f"{total_bytes / 1e6:.1f} MB in {elapsed:.1f}s ({mib_s:.1f} MiB/s)"
        )
        if errors:
            print(f"{len(errors)} download errors (continuing):")
            for name, err in errors[:10]:
                print(f"  - {name}: {err}")
            if len(errors) > 10:
                print(f"  … {len(errors) - 10} more")

        if not files:
            print("No files downloaded successfully; aborting.")
            return 1

        # Gather all files (don’t rely on extension)
        archive_files: List[Path] = []
        for root, _, fs in os.walk(tempdir):
            r = Path(root)
            for f in fs:
                archive_files.append(r / f)

        # ---- Load JSON → records ----
        posts = load_post_records(archive_files)
        print(f"Total JSON records loaded: {len(posts)}")

        # Filter to dict posts only; log non-dicts
        clean_posts = [p for p in posts if isinstance(p, dict)]
        if len(clean_posts) != len(posts):
            print(f"Filtered out {len(posts) - len(clean_posts)} non-dict records.")

        if not clean_posts:
            print("No valid post dicts; aborting push.")
            return 1

        df = pd.DataFrame(clean_posts)

        # Basic metrics
        if "id" in df.columns:
            print(f"Rows: {len(df)}, Unique posts by id: {df['id'].nunique()}")
        else:
            print("Warning: no 'id' column found; pushing without dedup by id.")

        if "comments" in df.columns:
            df = df.sort_values(
                "comments",
                key=lambda s: s.apply(
                    lambda v: len(v) if isinstance(v, (list, tuple)) else 0
                ),
                ascending=False,
            )

        if "id" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset="id", keep="first")
            print(f"Deduplicated by id: {before} → {len(df)}")

        dataset = Dataset.from_pandas(df, preserve_index=False)

        try:
            commit_info = dataset.push_to_hub(
                "Nech-C/reddit-sentiment",
                private=False,
                token=hf_token,
            )
            print(
                f"Pushed successfully: {commit_info.commit_url} (oid={commit_info.oid})"
            )

            if delete_after_upload:
                # Optional: parallelize deletions, too (they’re fast, but why not)
                del_errors: List[Tuple[str, str]] = []

                def _delete_one(b: storage.Blob) -> Tuple[str, Exception | None]:
                    try:
                        b.delete()
                        return b.name, None
                    except Exception as e:
                        return b.name, e

                with ThreadPoolExecutor(max_workers=min(32, len(blobs) or 1)) as ex:
                    futures = [ex.submit(_delete_one, b) for b in blobs]
                    for fut in as_completed(futures):
                        name, err = fut.result()
                        if err:
                            del_errors.append((name, repr(err)))

                print(
                    f"Deleted {len(blobs) - len(del_errors)}/{len(blobs)} blobs from {bucket_name}"
                )
                if del_errors:
                    print(f"{len(del_errors)} deletion errors:")
                    for name, err in del_errors[:10]:
                        print(f"  - {name}: {err}")
                    if len(del_errors) > 10:
                        print(f"  … {len(del_errors) - 10} more")

        except Exception as e:
            print(f"Upload failed or verification failed: {e!r}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
