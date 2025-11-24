from io import BytesIO
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import pandas as pd

bucket = "amzn-dataset-bucket"
prefix = "parquet_chunks/"
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def load_first_parquet_df() -> pd.DataFrame:
    """Fetch the first parquet object under the prefix and return it as a DataFrame."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                print(f"Loading s3://{bucket}/{key}")
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                return pd.read_parquet(BytesIO(body))
    raise RuntimeError(f"No parquet files found under s3://{bucket}/{prefix}")


df = load_first_parquet_df()

# Examine the structure of the data
print("Examining data structure:")
print("=" * 50)

# Look at the first row to understand the structure
first_row = df.iloc[0]
print("First row details:")
print(f"ID: {first_row['id']}")

print(f"Context features: {first_row['context_features']}")
print(f"Type: {type(first_row['context_features'])}")

print(f"Items features (first 3 items): {first_row['items_features'][:3]}")
print(f"Type: {type(first_row['items_features'])}")

print(f"Items dense features (first 2 arrays): {first_row['items_dense_features'][:2]}")
print(f"Type: {type(first_row['items_dense_features'])}")

print(f"Items dense features 2 (first 2 arrays): {first_row['items_dense_features2'][:2]}")
print(f"Type: {type(first_row['items_dense_features2'])}")

print(f"Labels (first 10): {first_row['labels'][:10]}")
print(f"Type: {type(first_row['labels'])}")

# Check lengths
print("\nLength information for first row:")
print(f"Items features length: {len(first_row['items_features'])}")
print(f"Items dense features length: {len(first_row['items_dense_features'])}")
print(f"Items dense features 2 length: {len(first_row['items_dense_features2'])}")
print(f"Labels length: {len(first_row['labels'])}")

# Check if all rows have consistent lengths
print("\nChecking consistency across rows:")
items_lengths = df['items_features'].apply(len)
dense_lengths = df['items_dense_features'].apply(len)
dense2_lengths = df['items_dense_features2'].apply(len)
labels_lengths = df['labels'].apply(len)

print(f"Items features lengths - min: {items_lengths.min()}, max: {items_lengths.max()}")
print(f"Dense features lengths - min: {dense_lengths.min()}, max: {dense_lengths.max()}")
print(f"Dense features 2 lengths - min: {dense2_lengths.min()}, max: {dense2_lengths.max()}")
print(f"Labels lengths - min: {labels_lengths.min()}, max: {labels_lengths.max()}")
