"""S3 utilities for parquet chunk management.

Provides download and upload functionality for the Alibaba PRM dataset.
"""

import os
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config

S3_BUCKET = "amzn-dataset-bucket"
S3_PREFIX = "parquet_chunks/"
LOCAL_DIR = "parquet_chunks"


def download_all_chunks(output_dir: str = LOCAL_DIR) -> dict:
    """Download all Parquet chunks from S3 to local directory.

    Args:
        output_dir: Local directory to save chunks

    Returns:
        Dict with 'downloaded', 'skipped', 'total' counts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Unsigned client for public bucket access
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    print(f"Downloading from s3://{S3_BUCKET}/{S3_PREFIX} to {output_path.absolute()}")

    # List all parquet files
    paginator = s3.get_paginator("list_objects_v2")
    all_keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                all_keys.append(obj["Key"])

    total_files = len(all_keys)
    print(f"Found {total_files} Parquet files")

    downloaded, skipped = 0, 0
    for idx, key in enumerate(all_keys, start=1):
        filename = os.path.basename(key)
        local_path = output_path / filename

        if local_path.exists():
            skipped += 1
            continue

        try:
            s3.download_file(S3_BUCKET, key, str(local_path))
            downloaded += 1
            if idx == 1 or idx % 25 == 0 or idx == total_files:
                print(f"[{idx}/{total_files}] Downloaded {filename}")
        except Exception as e:
            print(f"[{idx}/{total_files}] Failed: {filename} - {e}")

    print(f"Done: {downloaded} downloaded, {skipped} skipped")
    return {"downloaded": downloaded, "skipped": skipped, "total": total_files}


def delete_s3_prefix(bucket: str = S3_BUCKET, prefix: str = S3_PREFIX) -> int:
    """Delete all objects under the given S3 prefix.

    Args:
        bucket: S3 bucket name
        prefix: S3 prefix to delete

    Returns:
        Number of objects deleted
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))

    if not keys:
        print(f"No objects found under {prefix}")
        return 0

    print(f"Deleting {len(keys)} objects under {prefix}...")
    deleted = 0
    for start in range(0, len(keys), 1000):
        batch = keys[start:start + 1000]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in batch]})
        deleted += len(batch)
        print(f"Deleted {deleted}/{len(keys)}")
    return deleted


def upload_parquet_dir(
    local_dir: str = LOCAL_DIR,
    bucket: str = S3_BUCKET,
    prefix: str = S3_PREFIX,
) -> int:
    """Upload all parquet files from local directory to S3.

    Args:
        local_dir: Local directory containing parquet files
        bucket: S3 bucket name
        prefix: S3 prefix for uploads

    Returns:
        Number of files uploaded
    """
    local = Path(local_dir)
    if not local.is_dir():
        raise FileNotFoundError(f"Directory not found: {local}")

    files = sorted(local.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {local}")

    total_gb = sum(f.stat().st_size for f in files) / (1024**3)
    print(f"Uploading {len(files)} files ({total_gb:.2f} GB) to s3://{bucket}/{prefix}")

    s3 = boto3.client("s3")
    for idx, path in enumerate(files, start=1):
        key = f"{prefix}{path.relative_to(local).as_posix()}"
        s3.upload_file(str(path), bucket, key)
        if idx == 1 or idx % 25 == 0 or idx == len(files):
            print(f"[{idx}/{len(files)}] Uploaded {path.name}")
    return len(files)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python s3_utils.py [download|upload]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "download":
        download_all_chunks()
    elif cmd == "upload":
        delete_s3_prefix()
        upload_parquet_dir()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
