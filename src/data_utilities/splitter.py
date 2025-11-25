"""

This module splits the Alibaba dataset into "Training" and "Testing" sets.

It uses Deterministic Hashing instead of random shuffling to ensure every Arm (Linear, Neural, Random) sees the exact same 
users in the exact same order. This guarantees a fair comparison.
"""

from pathlib import Path
from typing import Dict, Literal, Any
import numpy as np
import pandas as pd
from sklearn.utils import murmurhash3_32


# Local parquet directory - relative to project root
# (Goes up 3 levels from src/data/splitter.py to find parquet_chunks)
LOCAL_PARQUET_DIR = Path(__file__).parent.parent.parent / "parquet_chunks"


def _resolve_local_chunk_path(chunk_key: str, base_dir: Path = LOCAL_PARQUET_DIR) -> Path:
    """Find the actual file path on your computer.

    Allows you to ask for 'chunk_0.parquet' without knowing exactly 
    where the folder is located on the disk.
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


def hash_record_id(record_id: int, seed: int = 42) -> int:
    """Turn a Request ID into a consistent number between 0 and 99.

    This acts like a fingerprint.
    - ID 100 with Seed 42 will ALWAYS return the same number.
    - This allows us to consistently assign specific users to Train or Test.
    """
    return murmurhash3_32(record_id, seed=seed, positive=True) % 100


def split_dataframe(
    df: pd.DataFrame,
    split: Literal['train', 'test', 'all'] = 'all',
    train_ratio: float = 0.8,
    seed: int = 42,
    id_column: str = 'id'
) -> pd.DataFrame:
    """Filter the data into a specific split.

    Logic:
    1. Calculate the fingerprint (0-99) for every row's ID.
    2. If fingerprint < 80 (assuming 80% train), it goes to Training.
    3. Otherwise, it goes to Testing.
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


def load_chunk_split(
    chunk_path: str,
    split: Literal['train', 'test', 'all'] = 'all',
    train_ratio: float = 0.8,
    seed: int = 42
) -> pd.DataFrame:
    """Load a file and immediately filter it to the requested split.

    Combines loading and splitting into one step for convenience.
    """
    local_path = _resolve_local_chunk_path(chunk_path)
    df = pd.read_parquet(local_path)
    return split_dataframe(df, split=split, train_ratio=train_ratio, seed=seed)


def get_split_statistics(
    df: pd.DataFrame,
    split: Literal['train', 'test'] = 'train',
    train_ratio: float = 0.8,
    seed: int = 42,
    id_column: str = 'id',
    label_column: str = 'labels'
) -> Dict[str, Any]:
    """Health Check: Calculate basic stats for a split.

    Verifies how many rows and clicks ended up in this split.
    Useful to ensure the Train/Test split is balanced (e.g., similar CTR in both).
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


def validate_split_reproducibility(
    chunk_path: str,
    seed: int = 42,
    num_trials: int = 3
) -> bool:
    """Sanity Check: Ensure the split logic is actually deterministic.

    Runs the split multiple times and confirms the results are identical.
    Returns True if everything is working correctly.
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
