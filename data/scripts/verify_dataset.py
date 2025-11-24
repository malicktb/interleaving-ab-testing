from io import BytesIO
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import numpy as np
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

print("DATASET DESCRIPTION VERIFICATION")
print("=" * 50)

# 1. Check basic structure
print("1. BASIC STRUCTURE")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

# 2. Verify request ID
print("2. REQUEST ID")
print(f"Type: {df['id'].dtype}")
print(f"Sample values: {df['id'].head(3).tolist()}")
print()

# 3. Verify user category features (context_features)
print("3. USER CATEGORY FEATURES (context_features)")
sample_context = df['context_features'].iloc[0]
print(f"Type: {type(sample_context)}")
print(f"Shape: {sample_context.shape}")
print(f"Sample values: {sample_context}")
print()

# 4. Verify item category features (items_features)
print("4. ITEM CATEGORY FEATURES (items_features)")
sample_items = df['items_features'].iloc[0]
print(f"Type: {type(sample_items)}")
print(f"Length: {len(sample_items)}")
print(f"First item type: {type(sample_items[0])}")
print(f"First item values: {sample_items[0]}")
print(f"First 3 items: {sample_items[:3]}")
print()

# 5. Verify basic dense features (items_dense_features)
print("5. BASIC DENSE FEATURES (items_dense_features)")
sample_dense = df['items_dense_features'].iloc[0]
print(f"Type: {type(sample_dense)}")
print(f"Length: {len(sample_dense)}")
print(f"First array type: {type(sample_dense[0])}")
print(f"First array length: {len(sample_dense[0])}")
print(f"First array values: {sample_dense[0]}")
print()

# 6. Verify optional dense features (items_dense_features2)
print("6. OPTIONAL DENSE FEATURES (items_dense_features2)")
sample_dense2 = df['items_dense_features2'].iloc[0]
print(f"Type: {type(sample_dense2)}")
print(f"Length: {len(sample_dense2)}")
print(f"First array type: {type(sample_dense2[0])}")
print(f"First array length: {len(sample_dense2[0])}")
print(f"First array values: {sample_dense2[0]}")
print()

# 7. Verify labels
print("7. LABELS")
sample_labels = df['labels'].iloc[0]
print(f"Type: {type(sample_labels)}")
print(f"Length: {len(sample_labels)}")
print(f"Sample values: {sample_labels[:10]}")
print(f"Unique values: {np.unique(sample_labels)}")
print()

# 8. Verify consistency across all rows
print("8. CONSISTENCY CHECK")
print(f"All rows have same number of items: {len(df['items_features'].apply(len).unique()) == 1}") 
items_count = df['items_features'].apply(len)
print(f"Items count per row: min={items_count.min()}, max={items_count.max()}")

dense_count = df['items_dense_features'].apply(len)
print(f"Dense features count per row: min={dense_count.min()}, max={dense_count.max()}")

dense2_count = df['items_dense_features2'].apply(len)
print(f"Optional dense features count per row: min={dense2_count.min()}, max={dense2_count.max()}")

labels_count = df['labels'].apply(len)
print(f"Labels count per row: min={labels_count.min()}, max={labels_count.max()}")

# Check array lengths within items
def get_inner_array_lengths(series):
    return series.apply(lambda x: [len(arr) for arr in x])

dense_lengths = get_inner_array_lengths(df['items_dense_features'])
dense2_lengths = get_inner_array_lengths(df['items_dense_features2'])

print(f"Basic dense feature array lengths: {np.unique(np.concatenate(dense_lengths.values))}")
print(f"Optional dense feature array lengths: {np.unique(np.concatenate(dense2_lengths.values))}")
