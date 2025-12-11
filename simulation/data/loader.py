"""Data loading utilities for H-MDB experiments.

Loads Yahoo LTR dataset from Parquet format and provides
iteration over train/test splits.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Generator
import numpy as np
import pyarrow.parquet as pq

from core.types import QueryRecord


# Required columns for valid parquet chunks
REQUIRED_COLUMNS = {"query_id", "features", "relevance"}


class DataLoader:
    """Load Yahoo LTR dataset from Parquet files.

    Provides train/test iteration and sampling functionality.
    """

    def __init__(
        self,
        data_dir: str = "data/processed/yahoo_parquet/set1",
        train_ratio: float = 0.8,
        seed: int = 42,
        num_chunks: int = None,
    ):
        """Initialize data loader.

        Args:
            data_dir: Path to data directory with train/test subdirs.
            train_ratio: Fraction of data for training (default 0.8).
            seed: Random seed for sampling.
            num_chunks: Optional limit on number of chunks to load.
        """
        self.data_dir = Path(data_dir)
        self.train_ratio = train_ratio
        self.seed = seed
        self.num_chunks = num_chunks

        self.train_chunks = self.discover_chunks("train")
        self.test_chunks = self.discover_chunks("test")

    def discover_chunks(self, split: str) -> List[Path]:
        """Discover parquet chunks for a split.

        Args:
            split: "train" or "test".

        Returns:
            List of paths to chunk files.
        """
        split_dir = self.data_dir / split
        if not split_dir.exists():
            return []

        chunks = sorted(split_dir.glob("chunk_*.parquet"))
        if self.num_chunks is not None:
            chunks = chunks[:self.num_chunks]
        return chunks

    def load_chunk(self, chunk_path: Path) -> Generator[QueryRecord, None, None]:
        """Load a parquet chunk and yield QueryRecord instances.

        Args:
            chunk_path: Path to the parquet chunk file.

        Yields:
            QueryRecord instances for each query in the chunk.

        Raises:
            ValueError: If required columns are missing from the chunk.
        """
        table = pq.read_table(chunk_path)

        # Validate required columns exist
        available_columns = set(table.column_names)
        missing_columns = REQUIRED_COLUMNS - available_columns
        if missing_columns:
            raise ValueError(
                f"Chunk {chunk_path} missing required columns: {missing_columns}. "
                f"Available: {available_columns}"
            )

        query_ids = table.column("query_id").to_pylist()
        features_col = table.column("features").to_pylist()
        relevance_col = table.column("relevance").to_pylist()

        # Batch convert to numpy arrays upfront for efficiency
        relevance_arrays = [np.asarray(rels, dtype=np.int8) for rels in relevance_col]
        feature_arrays = [np.asarray(feats, dtype=np.float32) for feats in features_col]

        # Yield records using pre-converted arrays
        for qid, features, relevance in zip(query_ids, feature_arrays, relevance_arrays):
            yield QueryRecord(
                query_id=str(qid),
                num_items=len(relevance),
                features=features,
                relevance=relevance,
            )

    def iter_records(self, chunks: List[Path]) -> Generator[QueryRecord, None, None]:
        """Iterate over all records in given chunks.

        Args:
            chunks: List of chunk paths.

        Yields:
            QueryRecord instances.
        """
        for chunk_path in chunks:
            yield from self.load_chunk(chunk_path)

    def iter_train_records(self) -> Generator[QueryRecord, None, None]:
        """Iterate over training records.

        Yields:
            QueryRecord instances from training set.
        """
        yield from self.iter_records(self.train_chunks)

    def iter_test_records(self) -> Generator[QueryRecord, None, None]:
        """Iterate over test records.

        Yields:
            QueryRecord instances from test set.
        """
        yield from self.iter_records(self.test_chunks)

    def iter_test_records_infinite(
        self,
        shuffle: bool = True,
        seed: int = None
    ) -> Generator[QueryRecord, None, None]:
        """Fix 2: Iterate over test records infinitely with optional shuffling.

        Cycles through the dataset repeatedly, allowing simulations to run
        for arbitrary lengths (e.g., 20,000 rounds with only 6,983 queries).

        Args:
            shuffle: Whether to shuffle chunk order each epoch (default True).
            seed: Random seed for shuffling (uses self.seed if not provided).

        Yields:
            QueryRecord instances, cycling indefinitely.
        """
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        epoch = 0

        while True:
            chunks = list(self.test_chunks)
            if shuffle:
                rng.shuffle(chunks)

            for chunk_path in chunks:
                yield from self.load_chunk(chunk_path)

            epoch += 1
            # Optional: print epoch progress for debugging
            # print(f"[DataLoader] Completed epoch {epoch}")

    def sample_train_records(
        self,
        sample_size: int,
        seed: int = None
    ) -> List[QueryRecord]:
        """Sample training records using reservoir sampling.

        Args:
            sample_size: Number of records to sample.
            seed: Random seed (defaults to self.seed if not provided).

        Returns:
            List of sampled QueryRecord objects.
        """
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        reservoir = []
        records_seen = 0

        for record in self.iter_train_records():
            records_seen += 1

            if len(reservoir) < sample_size:
                reservoir.append(record)
            else:
                j = rng.integers(0, records_seen)
                if j < sample_size:
                    reservoir[j] = record

        return reservoir
