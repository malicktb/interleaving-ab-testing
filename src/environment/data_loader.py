from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


@dataclass
class QueryRecord:
    query_id: str
    num_items: int
    features: np.ndarray
    relevance: np.ndarray


class DataLoader:

    def __init__(
        self,
        data_dir="data/processed/yahoo_parquet/set1",
        train_ratio=0.8,
        seed=42,
        num_chunks=None,
    ):
        self.data_dir = Path(data_dir)
        self.train_ratio = train_ratio
        self.seed = seed
        self.num_chunks = num_chunks

        self.train_chunks = self.discover_chunks("train")
        self.test_chunks = self.discover_chunks("test")

    def discover_chunks(self, split):
        split_dir = self.data_dir / split
        if not split_dir.exists():
            return []

        chunks = sorted(split_dir.glob("chunk_*.parquet"))
        if self.num_chunks is not None:
            chunks = chunks[:self.num_chunks]
        return chunks

    def load_chunk(self, chunk_path):
        table = pq.read_table(chunk_path)
        df = table.to_pandas()

        for _, row in df.iterrows():
            features_list = row["features"]
            features = np.array([np.array(f, dtype=np.float32) for f in features_list])
            relevance = np.array(row["relevance"], dtype=np.int8)

            yield QueryRecord(
                query_id=str(row["query_id"]),
                num_items=len(relevance),
                features=features,
                relevance=relevance,
            )

    def iter_records(self, chunks):
        for chunk_path in chunks:
            yield from self.load_chunk(chunk_path)

    def iter_train_records(self):
        yield from self.iter_records(self.train_chunks)

    def iter_test_records(self):
        yield from self.iter_records(self.test_chunks)

    @property
    def train_chunk_count(self):
        return len(self.train_chunks)

    @property
    def test_chunk_count(self):
        return len(self.test_chunks)

    def sample_train_records(self, sample_size, seed=42):
        rng = np.random.default_rng(seed)
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
