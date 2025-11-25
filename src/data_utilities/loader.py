"""
This module handles loading the Alibaba dataset. It ensures that every experiment gets the exact same 'Train' and 'Test' data (Reproducibility).
"""

from pathlib import Path
from typing import Iterator, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd

from .splitter import load_chunk_split


class ReservoirSampler:
    """A helper to collect a small random sample from a massive stream.

    We cannot train the Linear/MLP models on the full dataset (millions of rows)
    without running out of RAM. This class watches the data stream and keeps
    a fixed number of random records (e.g., 100,000) to represent the whole dataset.

    Attributes:
        sample_size: How many records to keep in memory.
        reservoir: The list of collected records.
        records_seen: Counter for how many rows have passed through.
    """

    def __init__(self, sample_size: int, seed: int = 42):
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)
        self.reservoir: List[dict] = []
        self.records_seen = 0

    def update(self, chunk_df: pd.DataFrame) -> None:
        """Look at a new chunk of data and randomly keep some rows."""
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

    def get_sample(self) -> pd.DataFrame:
        """Return the final collected sample."""
        return pd.DataFrame(self.reservoir)


class DataLoader:
    """Main interface for the Alibaba dataset.

    This class manages the raw Parquet files and splits them into 'Train'
    and 'Test' sets based on a hash of the Record ID. This guarantees fairness:
    every Agent sees the exact same data.

    Attributes:
        parquet_dir: Folder where the data files live.
        train_ratio: Percentage of data used for training (0.0 to 1.0).
        seed: Random seed for consistent splitting.
    """

    def __init__(
        self,
        parquet_dir: str = "parquet_chunks",
        train_ratio: float = 0.8,
        seed: int = 42,
        num_chunks: Optional[int] = None,
    ):
        self.parquet_dir = Path(parquet_dir)
        self.train_ratio = train_ratio
        self.seed = seed
        self.num_chunks = num_chunks
        self._chunk_files = self._discover_chunks()

    def _discover_chunks(self) -> List[Path]:
        """Find all .parquet files in the directory."""
        if not self.parquet_dir.exists():
            raise FileNotFoundError(f"Parquet directory not found: {self.parquet_dir}")

        chunks = sorted(self.parquet_dir.glob("chunk_*.parquet"))
        if not chunks:
            raise FileNotFoundError(f"No chunk files found in {self.parquet_dir}")

        if self.num_chunks is not None:
            chunks = chunks[: self.num_chunks]

        return chunks

    @property
    def chunk_count(self) -> int:
        """Total number of file chunks found."""
        return len(self._chunk_files)

    def stream_training_data(
        self,
        sample_size: int = 16667,
        seed: int = 42,
    ) -> Tuple[Iterator[pd.DataFrame], ReservoirSampler]:
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

        def _generator() -> Iterator[pd.DataFrame]:
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

    def iter_chunks(
        self, split: Literal["train", "test", "all"] = "all"
    ) -> Iterator[pd.DataFrame]:
        """Yield data in large blocks (DataFrames).
        
        Useful for batch processing when you don't need sampling.
        """
        for chunk_path in self._chunk_files:
            yield load_chunk_split(
                str(chunk_path),
                split=split,
                train_ratio=self.train_ratio,
                seed=self.seed,
            )

    def iter_records(
        self, split: Literal["train", "test", "all"] = "all"
    ) -> Iterator[dict]:
        """Yield data one row at a time (Dictionaries).
        
        Useful for the main Simulation Loop, where we simulate one user request
        at a time.
        """
        for chunk_df in self.iter_chunks(split):
            yield from chunk_df.to_dict("records")