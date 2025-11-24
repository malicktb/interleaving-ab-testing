from pathlib import Path
import boto3

bucket = "amzn-dataset-bucket"
prefix = "parquet_chunks/"
s3 = boto3.client("s3")

def delete_prefix(bucket: str, prefix: str) -> int:
    """Delete all objects under the given prefix (handles pagination) with progress."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))

    if not keys:
        print(f"No existing objects found under prefix {prefix!r}")
        return 0

    print(f"Deleting {len(keys)} objects under prefix {prefix!r}...")
    deleted = 0
    for start in range(0, len(keys), 1000):
        batch_keys = keys[start : start + 1000]
        s3.delete_objects(
            Bucket=bucket, Delete={"Objects": [{"Key": key} for key in batch_keys]}
        )
        deleted += len(batch_keys)
        print(f"Deleted {deleted}/{len(keys)}")

    return deleted


def upload_parquet_dir(bucket: str, prefix: str, local_dir: str) -> int:
    """Upload all .parquet files under local_dir to s3://bucket/prefix/ with progress."""
    local = Path(local_dir)
    if not local.is_dir():
        raise FileNotFoundError(f"Local directory not found: {local}")
    files = sorted(local.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found under: {local}")

    total_size_bytes = sum(path.stat().st_size for path in files)
    total_gib = total_size_bytes / (1024**3)
    print(f"Uploading {len(files)} parquet files ({total_gib:.2f} GiB) "f"to s3://{bucket}/{prefix}")

    uploaded = 0
    for idx, path in enumerate(files, start=1):
        key = f"{prefix}{path.relative_to(local).as_posix()}"
        s3.upload_file(str(path), bucket, key)
        uploaded += 1
        if idx == 1 or idx % 25 == 0 or idx == len(files):
            pct = (idx / len(files)) * 100
            print(f"Uploaded {idx}/{len(files)} ({pct:.1f}%) - {path.name}")
    return uploaded


if __name__ == "__main__":
    deleted = delete_prefix(bucket, prefix)
    print(f"Deleted {deleted} existing objects under {prefix!r}")
    uploaded = upload_parquet_dir(bucket, prefix, "parquet_chunks")
    print(f"Uploaded {uploaded} parquet files to s3://{bucket}/{prefix}")
