import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import argparse

def analyze_parquet_split(split_dir, max_chunks=None):
    split_path = Path(split_dir)
    chunks = sorted(split_path.glob("*.parquet"))

    if max_chunks:
        chunks = chunks[:max_chunks]

    if not chunks:
        return {"error": f"No parquet files found in {split_dir}"}

    total_queries = 0
    total_docs = 0
    docs_per_query = []
    relevance_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    feature_stats = {"min": float("inf"), "max": float("-inf"), "sum": 0, "count": 0}

    for chunk_path in chunks:
        table = pq.read_table(chunk_path)

        for row_idx in range(table.num_rows):
            relevance = table["relevance"][row_idx].as_py()
            features = table["features"][row_idx].as_py()

            total_queries += 1
            num_docs = len(relevance)
            total_docs += num_docs
            docs_per_query.append(num_docs)

            for rel in relevance:
                if rel in relevance_counts:
                    relevance_counts[rel] += 1

            for doc_features in features:
                for val in doc_features:
                    feature_stats["min"] = min(feature_stats["min"], val)
                    feature_stats["max"] = max(feature_stats["max"], val)
                    feature_stats["sum"] += val
                    feature_stats["count"] += 1

    return {
        "total_queries": total_queries,
        "total_docs": total_docs,
        "docs_per_query": {
            "min": min(docs_per_query) if docs_per_query else 0,
            "max": max(docs_per_query) if docs_per_query else 0,
            "mean": np.mean(docs_per_query) if docs_per_query else 0,
            "median": np.median(docs_per_query) if docs_per_query else 0,
        },
        "relevance_distribution": relevance_counts,
        "feature_range": {
            "min": feature_stats["min"],
            "max": feature_stats["max"],
            "mean": feature_stats["sum"] / feature_stats["count"] if feature_stats["count"] > 0 else 0,
        },
    }


def run_eda(data_dir="data/processed/yahoo_parquet", output_path=None):
    data_path = Path(data_dir)
    results = {}

    for set_dir in sorted(data_path.iterdir()):
        if not set_dir.is_dir():
            continue

        set_name = set_dir.name
        results[set_name] = {}

        for split in ["train", "valid", "test"]:
            split_dir = set_dir / split
            if split_dir.exists():
                print(f"Analyzing {set_name}/{split}...")
                results[set_name][split] = analyze_parquet_split(split_dir)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


def print_summary(results):
    print("\n" + "=" * 60)
    print("Yahoo LTR Dataset EDA Summary")
    print("=" * 60)

    for set_name, splits in results.items():
        print(f"\n{set_name.upper()}")
        print("-" * 40)

        for split_name, stats in splits.items():
            if "error" in stats:
                print(f"{split_name}: {stats['error']}")
                continue

            print(f"\n{split_name}:")
            print(f"Queries: {stats['total_queries']:,}")
            print(f"Documents: {stats['total_docs']:,}")
            print(f"Docs/Query: {stats['docs_per_query']['mean']:.1f} (min={stats['docs_per_query']['min']}, max={stats['docs_per_query']['max']})")

            print("Relevance Distribution:")
            for rel, count in sorted(stats['relevance_distribution'].items()):
                pct = count / stats['total_docs'] * 100 if stats['total_docs'] > 0 else 0
                print(f"Grade {rel}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/yahoo_parquet")
    parser.add_argument("--output", default="analysis/eda_results.json")

    args = parser.parse_args()

    results = run_eda(args.data_dir, args.output)
    print_summary(results)
