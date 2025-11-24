from io import BytesIO
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import pandas as pd
import pyarrow.parquet as pq

bucket = "amzn-dataset-bucket"
prefix = "parquet_chunks/"
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def fetch_first_parquet_bytes() -> bytes:
    """Retrieve the first parquet object under the prefix from S3."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                print(f"Downloading s3://{bucket}/{key}")
                return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    raise RuntimeError(f"No parquet files found under s3://{bucket}/{prefix}")


parquet_bytes = fetch_first_parquet_bytes()
parquet_file = pq.ParquetFile(BytesIO(parquet_bytes))
print("Parquet file metadata:")
print(f"Number of rows: {parquet_file.metadata.num_rows}")
print(f"Number of columns: {parquet_file.metadata.num_columns}")
print(f"Number of row groups: {parquet_file.metadata.num_row_groups}")

# Read the parquet file into a DataFrame
df = pd.read_parquet(BytesIO(parquet_bytes))

print("\nDataFrame info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Display basic info about each column
print("\nColumn information:")
print(df.info())

# Show first few rows
print("\nFirst 2 rows:")
print(df.head(2))

# Show data types
print("\nData types:")
print(df.dtypes)
