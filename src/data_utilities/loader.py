"""
This module handles loading the Alibaba dataset. It ensures that every experiment gets the exact same 'Train' and 'Test' data (Reproducibility).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .splitter import load_chunk_split


class ReservoirSampler:
    """A helper to collect a small random sample from a massive stream.

    We cannot train batch models (e.g., LinearArm) on the full dataset (millions
    of rows) without running out of RAM. This class watches the data stream and
    keeps a fixed number of random records to represent the whole dataset.

    Attributes:
        sample_size: How many records to keep in memory.
        reservoir: The list of collected records.
        records_seen: Counter for how many rows have passed through.
    """

    def __init__(self, sample_size, seed=42):
        """Initialize the reservoir sampler.

        Args:
            sample_size: Maximum number of records to keep.
            seed: Seed for random number generator.
        """
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)
        self.reservoir = []
        self.records_seen = 0

    def update(self, chunk_df):
        """Look at a new chunk of data and randomly keep some rows.

        Args:
            chunk_df: DataFrame chunk to process.
        """
        for idx in range(len(chunk_df)):
            self.records_seen += 1
            record = chunk_df.iloc[idx].to_dict()

            if len(self.reservoir) < self.sample_size:
                # If we haven't filled the buffer, just add it.
                self.reservoir.append(record)
            else:
                # If buffer is full, randomly replace an old item with this new one.
                # This mathematically guarantees a uniform random sample.
                j = self.rng.integers(0, self.records_seen)
                if j < self.sample_size:
                    self.reservoir[j] = record

    def get_sample(self):
        """Return the final collected sample.

        Returns:
            DataFrame containing the reservoir sample.
        """
        return pd.DataFrame(self.reservoir)


class DataLoader:
    """Main interface for loading parquet dataset chunks.

    This class manages the raw Parquet files and splits them into 'Train'
    and 'Test' sets based on a hash of the Record ID. This guarantees fairness:
    every Arm sees the exact same data.

    Attributes:
        parquet_dir: Folder where the data files live.
        train_ratio: Percentage of data used for training (0.0 to 1.0).
        seed: Random seed for consistent splitting.
    """

    def __init__(self, parquet_dir="parquet_chunks", train_ratio=0.8, seed=42, num_chunks=None):
        """Initialize the data loader.

        Args:
            parquet_dir: Directory containing parquet chunk files.
            train_ratio: Fraction of data to use for training (0.0 to 1.0).
            seed: Seed for deterministic splitting.
            num_chunks: Optional limit on number of chunks to load (None for all).
        """
        self.parquet_dir = Path(parquet_dir)
        self.train_ratio = train_ratio
        self.seed = seed
        self.num_chunks = num_chunks
        self._chunk_files = self._discover_chunks()

    def _discover_chunks(self):
        """Find all .parquet files in the directory.

        Returns:
            List of Path objects for each chunk file.
        """
        if not self.parquet_dir.exists():
            raise FileNotFoundError(f"Parquet directory not found: {self.parquet_dir}")

        chunks = sorted(self.parquet_dir.glob("chunk_*.parquet"))
        if not chunks:
            raise FileNotFoundError(f"No chunk files found in {self.parquet_dir}")

        if self.num_chunks is not None:
            chunks = chunks[: self.num_chunks]

        return chunks

    @property
    def chunk_count(self):
        """Total number of file chunks found."""
        return len(self._chunk_files)

    def stream_training_data(self, sample_size=16667, seed=42):
        """Read training data for different types of Arms.

        This solves a specific problem:
        1. Popularity Arm needs to see *every* row, but can forget them immediately.
        2. Linear Arm needs to see a *representative sample*, but must keep it in memory.

        This function returns both:
        - An iterator that yields all data chunk-by-chunk.
        - A sampler that automatically collects a random subset in the background.

        Args:
            sample_size: How many records to collect for the Linear Arm.
            seed: Seed for the sampler.

        Returns:
            (chunk_iterator, populated_sampler)
        """
        sampler = ReservoirSampler(sample_size=sample_size, seed=seed)

        def _generator():
            for i, chunk_path in enumerate(self._chunk_files):
                # Load only the 'Train' rows from this file
                chunk_df = load_chunk_split(
                    str(chunk_path),
                    split="train",
                    train_ratio=self.train_ratio,
                    seed=self.seed,
                )
                
                # Feed the sampler
                sampler.update(chunk_df)
                
                # Logging progress
                if (i + 1) % 50 == 0 or (i + 1) == len(self._chunk_files):
                    print(f"  Processed {i + 1}/{len(self._chunk_files)} chunks...")
                
                # Yield to the simulation loop
                yield chunk_df

        return _generator(), sampler

    def iter_chunks(self, split="all"):
        """Yield data in large blocks (DataFrames).

        Args:
            split: Which split to load ('train', 'test', or 'all').

        Yields:
            DataFrame for each chunk.
        """
        for chunk_path in self._chunk_files:
            yield load_chunk_split(
                str(chunk_path),
                split=split,
                train_ratio=self.train_ratio,
                seed=self.seed,
            )

    def iter_records(self, split="all"):
        """Yield data one row at a time (Dictionaries).

        Args:
            split: Which split to load ('train', 'test', or 'all').

        Yields:
            Dictionary for each record.
        """
        for chunk_df in self.iter_chunks(split):
            yield from chunk_df.to_dict("records")