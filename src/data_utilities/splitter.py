"""

This module splits the Alibaba dataset into "Training" and "Testing" sets.

It uses Deterministic Hashing instead of random shuffling to ensure every Arm (Linear, Popularity, Random) sees the exact same
users in the exact same order. This guarantees a fair comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import murmurhash3_32


# Local parquet directory - relative to project root
# (Goes up 3 levels from src/data_utilities/splitter.py to find parquet_chunks)
LOCAL_PARQUET_DIR = Path(__file__).parent.parent.parent / "parquet_chunks"


def _resolve_local_chunk_path(chunk_key, base_dir=LOCAL_PARQUET_DIR):
    """Find the actual file path on your computer.

    Args:
        chunk_key: File name or partial path to look up.
        base_dir: Base directory to search in.

    Returns:
        Path object pointing to the found file.
    """
    path = Path(chunk_key)

    if path.exists():
        return path

    # Check inside the base directory
    candidates = [base_dir / path, base_dir / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find file: {chunk_key}")


def hash_record_id(record_id, seed=42):
    """Turn a Request ID into a consistent number between 0 and 99.

    Args:
        record_id: The ID to hash.
        seed: Seed for the hash function.

    Returns:
        Integer from 0 to 99 representing the hash bucket.
    """
    return murmurhash3_32(record_id, seed=seed, positive=True) % 100


def split_dataframe(df, split='all', train_ratio=0.8, seed=42, id_column='id'):
    """Filter the data into a specific split.

    Args:
        df: DataFrame to split.
        split: Which split to return ('train', 'test', or 'all').
        train_ratio: Fraction of data for training (0.0 to 1.0).
        seed: Seed for deterministic hashing.
        id_column: Column name containing record IDs.

    Returns:
        DataFrame containing only the requested split.
    """
    if split == 'all':
        return df.copy()

    threshold = int(train_ratio * 100)
    
    # Create a True/False list based on the ID fingerprints
    mask = df[id_column].apply(lambda x: hash_record_id(x, seed) < threshold)

    if split == 'train':
        return df[mask].copy()
    else:  # split == 'test'
        return df[~mask].copy()


def load_chunk_split(chunk_path, split='all', train_ratio=0.8, seed=42):
    """Load a file and immediately filter it to the requested split.

    Args:
        chunk_path: Path or name of the parquet file.
        split: Which split to return ('train', 'test', or 'all').
        train_ratio: Fraction of data for training (0.0 to 1.0).
        seed: Seed for deterministic hashing.

    Returns:
        DataFrame containing only the requested split.
    """
    local_path = _resolve_local_chunk_path(chunk_path)
    df = pd.read_parquet(local_path)
    return split_dataframe(df, split=split, train_ratio=train_ratio, seed=seed)


def get_split_statistics(df, split='train', train_ratio=0.8, seed=42, id_column='id', label_column='labels'):
    """Health Check: Calculate basic stats for a split.

    Args:
        df: DataFrame to analyze.
        split: Which split to analyze ('train', 'test', or 'all').
        train_ratio: Fraction of data for training (0.0 to 1.0).
        seed: Seed for deterministic hashing.
        id_column: Column name containing record IDs.
        label_column: Column name containing labels.

    Returns:
        Dictionary with split, num_records, num_items, num_clicks, ctr, records_with_clicks.
    """
    split_df = split_dataframe(df, split=split, train_ratio=train_ratio, seed=seed, id_column=id_column)

    # Combine all labels into one big list to count them
    all_labels = np.concatenate(split_df[label_column].values)
    num_positive = np.sum(all_labels == 1.0)
    num_negative = np.sum(all_labels == 0.0)
    total_items = len(all_labels)

    # Calculate Click-Through Rate (CTR)
    ctr = num_positive / total_items if total_items > 0 else 0.0

    # Count how many user requests resulted in at least one click
    records_with_clicks = sum(np.any(labels == 1.0) for labels in split_df[label_column])

    return {
        'split': split,
        'num_records': len(split_df),
        'num_items': total_items,
        'num_clicks': int(num_positive),
        'ctr': ctr,
        'records_with_clicks': records_with_clicks
    }


def validate_split_reproducibility(chunk_path, seed=42, num_trials=3):
    """Sanity Check: Ensure the split logic is actually deterministic.

    Args:
        chunk_path: Path to the chunk file to test.
        seed: Seed for the hash function.
        num_trials: Number of times to repeat the split.

    Returns:
        True if all trials produced identical splits, False otherwise.
    """
    train_ids_list = []
    test_ids_list = []

    for _ in range(num_trials):
        train_df = load_chunk_split(chunk_path, split='train', seed=seed)
        test_df = load_chunk_split(chunk_path, split='test', seed=seed)

        train_ids_list.append(set(train_df['id'].values))
        test_ids_list.append(set(test_df['id'].values))

    # Check that every trial produced the exact same set of IDs
    train_consistent = all(ids == train_ids_list[0] for ids in train_ids_list)
    test_consistent = all(ids == test_ids_list[0] for ids in test_ids_list)

    return train_consistent and test_consistent
