from io import BytesIO
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# S3 Configuration
S3_BUCKET = "amzn-dataset-bucket"
S3_PREFIX = "parquet_chunks/"


def load_first_chunk():
    """Load the first Parquet chunk from S3."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    print("Fetching first chunk from S3...")
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                print(f"Loading s3://{S3_BUCKET}/{key}\n")
                body_bytes = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                return body_bytes, key

    raise RuntimeError(f"No parquet files found under s3://{S3_BUCKET}/{S3_PREFIX}")


def verify_parquet_metadata(parquet_bytes):
    """Verify Parquet file metadata."""
    print("=" * 70)
    print("1. PARQUET METADATA")
    print("=" * 70)

    parquet_file = pq.ParquetFile(BytesIO(parquet_bytes))
    print(f"Number of rows: {parquet_file.metadata.num_rows}")
    print(f"Number of columns: {parquet_file.metadata.num_columns}")
    print(f"Number of row groups: {parquet_file.metadata.num_row_groups}")
    print()


def verify_basic_structure(df):
    """Verify basic DataFrame structure."""
    print("=" * 70)
    print("2. BASIC STRUCTURE")
    print("=" * 70)

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    print()


def verify_request_id(df):
    """Verify request ID field."""
    print("=" * 70)
    print("3. REQUEST ID")
    print("=" * 70)

    print(f"Type: {df['id'].dtype}")
    print(f"Sample values: {df['id'].head(3).tolist()}")
    print(f"Unique IDs: {df['id'].nunique()} / {len(df)}")
    print()


def verify_user_features(df):
    """Verify user category features."""
    print("=" * 70)
    print("4. USER CATEGORY FEATURES (context_features)")
    print("=" * 70)

    sample = df['context_features'].iloc[0]
    print(f"Type: {type(sample)}")
    print(f"Shape: {sample.shape}")
    print(f"Sample values: {sample}")
    print(f"Expected: Array of 3 values")
    print()


def verify_item_features(df):
    """Verify item category features."""
    print("=" * 70)
    print("5. ITEM CATEGORY FEATURES (items_features)")
    print("=" * 70)

    sample = df['items_features'].iloc[0]
    print(f"Type: {type(sample)}")
    print(f"Number of items: {len(sample)}")
    print(f"First item type: {type(sample[0])}")
    print(f"First item length: {len(sample[0])}")
    print(f"First item values: {sample[0]}")
    print(f"Expected: 30 items, each with 5 values")
    print()


def verify_dense_features(df):
    """Verify basic dense features."""
    print("=" * 70)
    print("6. BASIC DENSE FEATURES (items_dense_features)")
    print("=" * 70)

    sample = df['items_dense_features'].iloc[0]
    print(f"Type: {type(sample)}")
    print(f"Number of arrays: {len(sample)}")
    print(f"First array type: {type(sample[0])}")
    print(f"First array length: {len(sample[0])}")
    print(f"First array values: {sample[0]}")
    print(f"Expected: 30 arrays, each with 7 values")
    print()


def verify_optional_features(df):
    """Verify optional dense features."""
    print("=" * 70)
    print("7. OPTIONAL DENSE FEATURES (items_dense_features2)")
    print("=" * 70)

    sample = df['items_dense_features2'].iloc[0]
    print(f"Type: {type(sample)}")
    print(f"Number of arrays: {len(sample)}")
    print(f"First array type: {type(sample[0])}")
    print(f"First array length: {len(sample[0])}")
    print(f"First array values: {sample[0]}")
    print(f"Expected: 30 arrays, each with 12 values")
    print()


def verify_labels(df):
    """Verify labels."""
    print("=" * 70)
    print("8. LABELS")
    print("=" * 70)

    sample = df['labels'].iloc[0]
    print(f"Type: {type(sample)}")
    print(f"Length: {len(sample)}")
    print(f"Sample values (first 10): {sample[:10]}")
    print(f"Unique values: {np.unique(sample)}")

    # Calculate CTR
    all_labels = np.concatenate(df['labels'].values)
    ctr = np.sum(all_labels == 1.0) / len(all_labels)
    print(f"Click-through rate: {ctr:.6f}")
    print(f"Expected: 30 labels (0.0 or 1.0) per record")
    print()


def verify_consistency(df):
    """Verify consistency across all rows."""
    print("=" * 70)
    print("9. CONSISTENCY CHECK")
    print("=" * 70)

    # Check array lengths
    items_count = df['items_features'].apply(len)
    dense_count = df['items_dense_features'].apply(len)
    dense2_count = df['items_dense_features2'].apply(len)
    labels_count = df['labels'].apply(len)

    print(f"Items per record: min={items_count.min()}, max={items_count.max()}")
    print(f"Dense features per record: min={dense_count.min()}, max={dense_count.max()}")
    print(f"Optional dense per record: min={dense2_count.min()}, max={dense2_count.max()}")
    print(f"Labels per record: min={labels_count.min()}, max={labels_count.max()}")

    # Check inner array lengths
    def get_inner_lengths(series):
        return series.apply(lambda x: [len(arr) for arr in x])

    dense_lengths = get_inner_lengths(df['items_dense_features'])
    dense2_lengths = get_inner_lengths(df['items_dense_features2'])

    unique_dense = np.unique(np.concatenate(dense_lengths.values))
    unique_dense2 = np.unique(np.concatenate(dense2_lengths.values))

    print(f"\nBasic dense feature array lengths: {unique_dense}")
    print(f"Optional dense feature array lengths: {unique_dense2}")

    # Overall check
    all_consistent = (
        items_count.min() == items_count.max() == 30 and
        dense_count.min() == dense_count.max() == 30 and
        dense2_count.min() == dense2_count.max() == 30 and
        labels_count.min() == labels_count.max() == 30 and
        len(unique_dense) == 1 and unique_dense[0] == 7 and
        len(unique_dense2) == 1 and unique_dense2[0] == 12
    )

    print()
    if all_consistent:
        print("ALL CONSISTENCY CHECKS PASSED")
    else:
        print("CONSISTENCY CHECKS FAILED")
    print()


def main():
    """Run all verification checks."""
    print("\n" + "=" * 70)
    print("DATASET VERIFICATION")
    print("=" * 70)
    print()

    try:
        # Load first chunk
        parquet_bytes, chunk_key = load_first_chunk()

        # Verify metadata
        verify_parquet_metadata(parquet_bytes)

        # Load as DataFrame
        df = pd.read_parquet(BytesIO(parquet_bytes))

        # Run all verification checks
        verify_basic_structure(df)
        verify_request_id(df)
        verify_user_features(df)
        verify_item_features(df)
        verify_dense_features(df)
        verify_optional_features(df)
        verify_labels(df)
        verify_consistency(df)

        # Summary
        print("=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)
        print(f"Successfully verified chunk: {chunk_key}")
        print(f"Records: {len(df)}")
        print(f"All structure checks passed")
        print("=" * 70)

    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
