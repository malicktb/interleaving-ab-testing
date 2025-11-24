import os
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.client import Config

S3_BUCKET = "amzn-dataset-bucket"
S3_PREFIX = "parquet_chunks/"
OUTPUT_DIR = "parquet_chunks"


def download_all_chunks():
    """Download all Parquet chunks from S3 to local directory."""
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    # Initialize S3 client (unsigned for public access)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    print("=" * 70)
    print("DOWNLOADING PARQUET CHUNKS FROM S3")
    print("=" * 70)
    print(f"\nSource: s3://{S3_BUCKET}/{S3_PREFIX}")
    print(f"Destination: {output_path.absolute()}")
    print()

    # List all parquet files in S3
    print("Fetching file list from S3...")
    paginator = s3.get_paginator("list_objects_v2")
    all_keys = []

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                all_keys.append(key)

    total_files = len(all_keys)
    print(f"Found {total_files} Parquet files\n")

    # Download each file
    downloaded = 0
    skipped = 0

    for idx, key in enumerate(all_keys, start=1):
        filename = os.path.basename(key)
        local_path = output_path / filename

        # Skip if file exists
        if local_path.exists():
            skipped += 1
            if idx % 100 == 0:
                print(f"[{idx}/{total_files}] Skipped {skipped} existing files...")
            continue

        # Download file
        try:
            s3.download_file(S3_BUCKET, key, str(local_path))
            downloaded += 1

            # Show progress every 25 files
            if idx == 1 or idx % 25 == 0 or idx == total_files:
                file_size_mb = local_path.stat().st_size / (1024 * 1024)
                progress_pct = (idx / total_files) * 100
                print(f"[{idx}/{total_files}] ({progress_pct:.1f}%) "
                      f"Downloaded {filename} ({file_size_mb:.1f} MB)")

        except Exception as e:
            print(f"[{idx}/{total_files}] ✗ Failed: {filename} - {e}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Downloaded: {downloaded}")
    print(f"Skipped: {skipped}")
    print(f"Total: {total_files}")

    if downloaded > 0 or skipped == total_files:
        total_size_bytes = sum(f.stat().st_size for f in output_path.glob("*.parquet"))
        total_size_gb = total_size_bytes / (1024 ** 3)
        print(f"\nTotal size: {total_size_gb:.2f} GB")
        print(f"Location: {output_path.absolute()}")

    if downloaded > 0:
        print("\nDownload complete!")
    elif skipped == total_files:
        print("\nAll files already exist locally.")

    print("=" * 70)


if __name__ == "__main__":
    download_all_chunks()
