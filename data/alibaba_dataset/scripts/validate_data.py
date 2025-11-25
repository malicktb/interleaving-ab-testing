"""Unified validation for Alibaba PRM dataset.

Combines:
- Dataset structure verification (parquet format, schema, consistency)
- Hash-based split validation (ratio, coverage, reproducibility)

Usage:
    python validate_data.py structure   # Verify parquet structure
    python validate_data.py split       # Validate train/test split
    python validate_data.py all         # Run all validations
"""

from io import BytesIO
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.client import Config

from src.data.splitter import (
    get_split_statistics,
    hash_record_id,
    load_chunk_split,
    validate_split_reproducibility,
)

S3_BUCKET = "amzn-dataset-bucket"
S3_PREFIX = "parquet_chunks/"

def load_first_chunk_bytes() -> tuple:
    """Load raw bytes of first parquet chunk from S3."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                print(f"Loading s3://{S3_BUCKET}/{key}")
                body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                return body, key

    raise RuntimeError(f"No parquet files under s3://{S3_BUCKET}/{S3_PREFIX}")


def verify_parquet_metadata(parquet_bytes: bytes) -> None:
    """Verify parquet file metadata."""
    print("\n" + "=" * 60)
    print("PARQUET METADATA")
    print("=" * 60)
    pf = pq.ParquetFile(BytesIO(parquet_bytes))
    print(f"Rows: {pf.metadata.num_rows}")
    print(f"Columns: {pf.metadata.num_columns}")
    print(f"Row groups: {pf.metadata.num_row_groups}")


def verify_schema(df: pd.DataFrame) -> None:
    """Verify DataFrame schema."""
    print("\n" + "=" * 60)
    print("SCHEMA")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")


def verify_consistency(df: pd.DataFrame) -> bool:
    """Verify array lengths are consistent across all rows."""
    print("\n" + "=" * 60)
    print("CONSISTENCY CHECK")
    print("=" * 60)

    items_len = df["items_features"].apply(len)
    dense_len = df["items_dense_features"].apply(len)
    dense2_len = df["items_dense_features2"].apply(len)
    labels_len = df["labels"].apply(len)

    print(f"items_features: {items_len.min()}-{items_len.max()} (expect 30)")
    print(f"items_dense_features: {dense_len.min()}-{dense_len.max()} (expect 30)")
    print(f"items_dense_features2: {dense2_len.min()}-{dense2_len.max()} (expect 30)")
    print(f"labels: {labels_len.min()}-{labels_len.max()} (expect 30)")

    # Check inner array dimensions
    dense_inner = np.unique([len(arr) for row in df["items_dense_features"] for arr in row])
    dense2_inner = np.unique([len(arr) for row in df["items_dense_features2"] for arr in row])

    print(f"Dense feature dims: {dense_inner} (expect [7])")
    print(f"Dense2 feature dims: {dense2_inner} (expect [12])")

    all_ok = (
        items_len.min() == items_len.max() == 30
        and dense_len.min() == dense_len.max() == 30
        and labels_len.min() == labels_len.max() == 30
        and list(dense_inner) == [7]
        and list(dense2_inner) == [12]
    )

    print(f"\n{'PASSED' if all_ok else 'FAILED'}")
    return all_ok


def verify_labels(df: pd.DataFrame) -> None:
    """Verify label statistics."""
    print("\n" + "=" * 60)
    print("LABELS")
    print("=" * 60)
    all_labels = np.concatenate(df["labels"].values)
    unique = np.unique(all_labels)
    ctr = np.sum(all_labels == 1.0) / len(all_labels)
    print(f"Unique values: {unique}")
    print(f"CTR: {ctr:.6f}")


def run_structure_validation() -> bool:
    """Run all structure validation checks."""
    print("\n" + "=" * 60)
    print("DATASET STRUCTURE VALIDATION")
    print("=" * 60)

    parquet_bytes, key = load_first_chunk_bytes()
    verify_parquet_metadata(parquet_bytes)

    df = pd.read_parquet(BytesIO(parquet_bytes))
    verify_schema(df)
    verify_labels(df)
    passed = verify_consistency(df)

    print(f"\nVerified: {key} ({len(df)} records)")
    return passed

def validate_hash_function() -> bool:
    """Validate hash function determinism and distribution."""
    print("\n" + "=" * 60)
    print("HASH FUNCTION")
    print("=" * 60)

    # Determinism
    test_id = 442358
    hashes = [hash_record_id(test_id, seed=42) for _ in range(10)]
    deterministic = len(set(hashes)) == 1
    print(f"Deterministic: {deterministic}")

    # Distribution
    sample_ids = range(1000, 11000)
    deciles = [0] * 10
    for rid in sample_ids:
        deciles[hash_record_id(rid, seed=42) // 10] += 1

    max_dev = max(abs(c - 1000) / 1000 for c in deciles)
    uniform = max_dev < 0.20
    print(f"Distribution max deviation: {max_dev*100:.1f}% (threshold 20%)")
    print(f"Uniform: {uniform}")

    return deterministic and uniform


def validate_split_ratio() -> bool:
    """Validate 80/20 split ratio."""
    print("\n" + "=" * 60)
    print("SPLIT RATIO")
    print("=" * 60)

    chunk_key = "parquet_chunks/chunk_0000.parquet"
    train = load_chunk_split(chunk_key, split="train", seed=42)
    test = load_chunk_split(chunk_key, split="test", seed=42)
    total = len(train) + len(test)

    ratio = len(train) / total
    print(f"Train: {len(train)} ({ratio*100:.1f}%)")
    print(f"Test: {len(test)} ({(1-ratio)*100:.1f}%)")

    passed = abs(ratio - 0.80) < 0.01
    print(f"Within 1% of 80/20: {passed}")
    return passed


def validate_coverage() -> bool:
    """Validate no overlap and complete coverage."""
    print("\n" + "=" * 60)
    print("COVERAGE")
    print("=" * 60)

    chunk_key = "parquet_chunks/chunk_0000.parquet"
    train = load_chunk_split(chunk_key, split="train", seed=42)
    test = load_chunk_split(chunk_key, split="test", seed=42)
    all_df = load_chunk_split(chunk_key, split="all", seed=42)

    train_ids = set(train["id"])
    test_ids = set(test["id"])
    all_ids = set(all_df["id"])

    overlap = train_ids & test_ids
    union = train_ids | test_ids

    print(f"Overlap: {len(overlap)} (expect 0)")
    print(f"Union matches all: {union == all_ids}")

    return len(overlap) == 0 and union == all_ids


def validate_label_distribution() -> bool:
    """Validate similar CTR between train and test."""
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION")
    print("=" * 60)

    train_pos, train_total = 0, 0
    test_pos, test_total = 0, 0

    for i in range(5):
        chunk_key = f"parquet_chunks/chunk_{i:04d}.parquet"
        df = load_chunk_split(chunk_key, split="all", seed=42)

        train_stats = get_split_statistics(df, split="train", seed=42)
        test_stats = get_split_statistics(df, split="test", seed=42)

        train_pos += train_stats["num_positive_labels"]
        train_total += train_stats["num_items"]
        test_pos += test_stats["num_positive_labels"]
        test_total += test_stats["num_items"]

    train_ctr = train_pos / train_total if train_total else 0
    test_ctr = test_pos / test_total if test_total else 0
    diff_pct = abs(train_ctr - test_ctr) / train_ctr * 100 if train_ctr else 0

    print(f"Train CTR: {train_ctr:.6f}")
    print(f"Test CTR: {test_ctr:.6f}")
    print(f"Difference: {diff_pct:.2f}% (threshold 5%)")

    return diff_pct < 5.0


def validate_reproducibility_check() -> bool:
    """Validate same seed produces same split."""
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY")
    print("=" * 60)

    chunk_key = "parquet_chunks/chunk_0000.parquet"
    result = validate_split_reproducibility(chunk_key, seed=42, num_trials=3)
    print(f"Reproducible across 3 trials: {result}")
    return result


def run_split_validation() -> bool:
    """Run all split validation checks."""
    print("\n" + "=" * 60)
    print("HASH-BASED SPLIT VALIDATION")
    print("=" * 60)

    results = {
        "hash_function": validate_hash_function(),
        "split_ratio": validate_split_ratio(),
        "coverage": validate_coverage(),
        "label_distribution": validate_label_distribution(),
        "reproducibility": validate_reproducibility_check(),
    }

    print("\n" + "=" * 60)
    print("SPLIT VALIDATION SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    return all(results.values())


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validate_data.py [structure|split|all]")
        sys.exit(1)

    cmd = sys.argv[1]
    passed = True

    if cmd in ("structure", "all"):
        passed = run_structure_validation() and passed
    if cmd in ("split", "all"):
        passed = run_split_validation() and passed

    if cmd not in ("structure", "split", "all"):
        print(f"Unknown command: {cmd}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED" if passed else "SOME VALIDATIONS FAILED")
    print("=" * 60)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
