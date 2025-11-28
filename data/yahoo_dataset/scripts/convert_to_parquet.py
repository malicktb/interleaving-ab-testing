import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = DATA_DIR / "parquet"

NUM_FEATURES = 700

CHUNK_SIZE = 5000


@dataclass
class QueryGroup:
    query_id: str
    features: list
    relevance: list


def parse_letor_line(line):
    parts = line.strip().split()
    if not parts:
        return None

    relevance = int(parts[0])
    qid = parts[1].split(":")[1]

    features = {}
    for part in parts[2:]:
        if ":" in part and not part.startswith("#"):
            fid_str, val_str = part.split(":")
            try:
                features[int(fid_str)] = float(val_str)
            except ValueError:
                continue

    return relevance, qid, features


def densify_features(sparse_dict, dim=NUM_FEATURES):
    dense = np.zeros(dim, dtype=np.float32)
    for fid, val in sparse_dict.items():
        if 1 <= fid <= dim:
            dense[fid - 1] = val
    return dense


def group_queries(filepath):
    current_qid = None
    current_features = []
    current_relevance = []

    with open(filepath, "r") as f:
        for line in f:
            result = parse_letor_line(line)
            if result is None:
                continue

            rel, qid, sparse_feats = result

            if qid != current_qid:
                if current_qid is not None:
                    yield QueryGroup(
                        query_id=current_qid,
                        features=current_features,
                        relevance=current_relevance,
                    )
                current_qid = qid
                current_features = []
                current_relevance = []

            current_features.append(densify_features(sparse_feats))
            current_relevance.append(rel)

        if current_qid is not None:
            yield QueryGroup(
                query_id=current_qid,
                features=current_features,
                relevance=current_relevance,
            )


def convert_file(input_path, output_dir, chunk_size=CHUNK_SIZE):
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        ("query_id", pa.string()),
        ("features", pa.list_(pa.list_(pa.float32()))),
        ("relevance", pa.list_(pa.int8())),
    ])

    stats = {
        "input_file": input_path.name,
        "total_queries": 0,
        "total_docs": 0,
        "chunks_written": 0,
    }

    chunk_idx = 0
    records = []

    print(f"  Converting {input_path.name}...", flush=True)
    start_time = time.time()

    for query_group in group_queries(input_path):
        record = {
            "query_id": query_group.query_id,
            "features": [f.tolist() for f in query_group.features],
            "relevance": query_group.relevance,
        }
        records.append(record)
        stats["total_queries"] += 1
        stats["total_docs"] += len(query_group.features)

        if len(records) >= chunk_size:
            table = pa.Table.from_pylist(records, schema=schema)
            chunk_path = output_dir / f"chunk_{chunk_idx:04d}.parquet"
            pq.write_table(table, chunk_path, compression="snappy")
            print(f"    Written chunk {chunk_idx}: {len(records)} queries", flush=True)
            chunk_idx += 1
            stats["chunks_written"] += 1
            records = []

    if records:
        table = pa.Table.from_pylist(records, schema=schema)
        chunk_path = output_dir / f"chunk_{chunk_idx:04d}.parquet"
        pq.write_table(table, chunk_path, compression="snappy")
        print(f"    Written chunk {chunk_idx}: {len(records)} queries", flush=True)
        stats["chunks_written"] += 1

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = elapsed
    print(f"  Done: {stats['total_queries']:,} queries, {stats['total_docs']:,} docs in {elapsed:.1f}s", flush=True)

    return stats


def validate_parquet(parquet_dir):
    results = {
        "valid": True,
        "total_files": 0,
        "total_queries": 0,
        "total_docs": 0,
        "feature_dim_check": True,
        "errors": [],
    }

    chunk_files = sorted(parquet_dir.glob("chunk_*.parquet"))
    results["total_files"] = len(chunk_files)

    for chunk_path in chunk_files:
        try:
            table = pq.read_table(chunk_path)
            df = table.to_pandas()

            for idx, row in df.iterrows():
                results["total_queries"] += 1
                num_docs = len(row["features"])
                results["total_docs"] += num_docs

                for features in row["features"]:
                    if len(features) != NUM_FEATURES:
                        results["valid"] = False
                        results["feature_dim_check"] = False
                        results["errors"].append(
                            f"Wrong feature dim in {chunk_path.name}, query {row['query_id']}"
                        )
                        break

                if len(row["relevance"]) != num_docs:
                    results["valid"] = False
                    results["errors"].append(
                        f"Relevance/features mismatch in {chunk_path.name}, query {row['query_id']}"
                    )

        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Error reading {chunk_path.name}: {str(e)}")

    return results


def main():
    print("=" * 60)
    print("Yahoo LTR Dataset - LETOR to Parquet Conversion")
    print("=" * 60)
    print()
    print(f"Feature dimension: {NUM_FEATURES}")
    print(f"Chunk size: {CHUNK_SIZE} queries")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    conversions = [
        ("set1_train", DATA_DIR / "set1.train.txt", OUTPUT_DIR / "set1" / "train"),
        ("set1_valid", DATA_DIR / "set1.valid.txt", OUTPUT_DIR / "set1" / "valid"),
        ("set1_test", DATA_DIR / "set1.test.txt", OUTPUT_DIR / "set1" / "test"),
        ("set2_train", DATA_DIR / "set2.train.txt", OUTPUT_DIR / "set2" / "train"),
        ("set2_valid", DATA_DIR / "set2.valid.txt", OUTPUT_DIR / "set2" / "valid"),
        ("set2_test", DATA_DIR / "set2.test.txt", OUTPUT_DIR / "set2" / "test"),
    ]

    all_stats = {}
    total_start = time.time()

    for name, input_path, output_dir in conversions:
        if not input_path.exists():
            print(f"WARNING: {input_path} not found, skipping...")
            continue

        print(f"\n{name}:")
        stats = convert_file(input_path, output_dir)
        all_stats[name] = stats

    total_elapsed = time.time() - total_start

    print()
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)

    total_queries = sum(s["total_queries"] for s in all_stats.values())
    total_docs = sum(s["total_docs"] for s in all_stats.values())
    total_chunks = sum(s["chunks_written"] for s in all_stats.values())

    print(f"Total queries:  {total_queries:,}")
    print(f"Total documents: {total_docs:,}")
    print(f"Total chunks:    {total_chunks}")
    print(f"Total time:      {total_elapsed:.1f}s")
    print()

    print("Validating Set 1...")
    for split in ["train", "valid", "test"]:
        split_dir = OUTPUT_DIR / "set1" / split
        if split_dir.exists():
            validation = validate_parquet(split_dir)
            status = "PASS" if validation["valid"] else "FAIL"
            print(f"  {split}: {status} ({validation['total_queries']:,} queries, {validation['total_docs']:,} docs)")
            if not validation["valid"]:
                for error in validation["errors"][:5]:
                    print(f"    ERROR: {error}")

    print()
    print(f"Output saved to: {OUTPUT_DIR}")
    print()

    print("Output file sizes:")
    for split in ["train", "valid", "test"]:
        split_dir = OUTPUT_DIR / "set1" / split
        if split_dir.exists():
            total_size = sum(f.stat().st_size for f in split_dir.glob("*.parquet"))
            print(f"  set1/{split}: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
