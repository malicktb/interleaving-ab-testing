import json
from collections import defaultdict
from pathlib import Path
import numpy as np


SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent

FILES = {
    "set1_train": DATA_DIR / "set1.train.txt",
    "set1_valid": DATA_DIR / "set1.valid.txt",
    "set1_test": DATA_DIR / "set1.test.txt",
    "set2_train": DATA_DIR / "set2.train.txt",
    "set2_valid": DATA_DIR / "set2.valid.txt",
    "set2_test": DATA_DIR / "set2.test.txt",
}


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


def analyze_file(filepath, sample_features=True):
    stats = {
        "total_docs": 0,
        "unique_queries": set(),
        "relevance_counts": defaultdict(int),
        "feature_ids": set(),
        "docs_per_query": defaultdict(int),
        "features_per_doc": [],
        "feature_value_samples": defaultdict(list),
    }

    max_feature_samples = 1000

    print(f"Analyzing {filepath.name}...", flush=True)

    with open(filepath, "r") as f:
        for line_num, line in enumerate(f):
            result = parse_letor_line(line)
            if result is None:
                continue

            rel, qid, features = result

            stats["total_docs"] += 1
            stats["unique_queries"].add(qid)
            stats["relevance_counts"][rel] += 1
            stats["feature_ids"].update(features.keys())
            stats["docs_per_query"][qid] += 1
            stats["features_per_doc"].append(len(features))

            if sample_features:
                for fid, val in features.items():
                    if len(stats["feature_value_samples"][fid]) < max_feature_samples:
                        stats["feature_value_samples"][fid].append(val)

            if (line_num + 1) % 100000 == 0:
                print(f"Processed {line_num + 1:,} lines...", flush=True)

    return stats


def compute_summary(stats):
    docs_per_query_values = list(stats["docs_per_query"].values())
    features_per_doc = stats["features_per_doc"]

    feature_stats = {}
    for fid, values in stats["feature_value_samples"].items():
        if values:
            arr = np.array(values)
            feature_stats[fid] = {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "samples": len(values),
            }

    return {
        "total_docs": stats["total_docs"],
        "unique_queries": len(stats["unique_queries"]),
        "relevance_distribution": dict(stats["relevance_counts"]),
        "feature_count": len(stats["feature_ids"]),
        "feature_id_min": min(stats["feature_ids"]) if stats["feature_ids"] else None,
        "feature_id_max": max(stats["feature_ids"]) if stats["feature_ids"] else None,
        "features_per_doc": {
            "min": min(features_per_doc) if features_per_doc else 0,
            "max": max(features_per_doc) if features_per_doc else 0,
            "mean": float(np.mean(features_per_doc)) if features_per_doc else 0,
            "std": float(np.std(features_per_doc)) if features_per_doc else 0,
        },
        "docs_per_query": {
            "min": min(docs_per_query_values) if docs_per_query_values else 0,
            "max": max(docs_per_query_values) if docs_per_query_values else 0,
            "mean": float(np.mean(docs_per_query_values)) if docs_per_query_values else 0,
            "std": float(np.std(docs_per_query_values)) if docs_per_query_values else 0,
        },
        "avg_sparsity": 1.0 - (np.mean(features_per_doc) / max(stats["feature_ids"])) if stats["feature_ids"] else 0,
    }


def main():
    print("=" * 60)
    print("Yahoo LTR Dataset - Exploratory Data Analysis")
    print("=" * 60)
    print()

    report = {}
    all_feature_ids = set()
    set1_feature_ids = set()
    set2_feature_ids = set()

    for name, path in FILES.items():
        if not path.exists():
            print(f"WARNING: {path} not found, skipping...")
            continue

        print(f"\nProcessing {name}...")
        stats = analyze_file(path)
        summary = compute_summary(stats)
        report[name] = summary

        all_feature_ids.update(stats["feature_ids"])
        if name.startswith("set1"):
            set1_feature_ids.update(stats["feature_ids"])
        else:
            set2_feature_ids.update(stats["feature_ids"])

        print(f"Done: {summary['total_docs']:,} docs, {summary['unique_queries']:,} queries")

    print("\nComputing global statistics...")

    intersection_features = set1_feature_ids & set2_feature_ids
    union_features = set1_feature_ids | set2_feature_ids

    report["global"] = {
        "total_feature_ids_union": len(union_features),
        "total_feature_ids_set1": len(set1_feature_ids),
        "total_feature_ids_set2": len(set2_feature_ids),
        "feature_ids_intersection": len(intersection_features),
        "feature_id_range": [min(all_feature_ids), max(all_feature_ids)] if all_feature_ids else [0, 0],
        "recommended_dense_dim": max(all_feature_ids) if all_feature_ids else 0,
    }

    total_relevance = defaultdict(int)
    total_docs = 0
    for name, summary in report.items():
        if name != "global" and "relevance_distribution" in summary:
            for rel, count in summary["relevance_distribution"].items():
                total_relevance[int(rel)] += count
                total_docs += count

    report["global"]["total_relevance_distribution"] = dict(total_relevance)
    report["global"]["total_docs_all_files"] = total_docs

    output_path = DATA_DIR / "eda_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print()
    print("=" * 60)
    print("EDA Summary")
    print("=" * 60)
    print()
    print(f"Total documents (all files): {total_docs:,}")
    print(f"Feature dimension (union): {report['global']['total_feature_ids_union']}")
    print(f"Feature ID range: {report['global']['feature_id_range']}")
    print(f"Set 1 features: {report['global']['total_feature_ids_set1']}")
    print(f"Set 2 features: {report['global']['total_feature_ids_set2']}")
    print(f"Intersection: {report['global']['feature_ids_intersection']}")
    print()
    print("Relevance Distribution (all files):")
    for rel in sorted(total_relevance.keys()):
        count = total_relevance[rel]
        pct = 100 * count / total_docs if total_docs > 0 else 0
        print(f"Grade {rel}: {count:,} ({pct:.1f}%)")
    print()

    print("Per-File Statistics:")
    print("-" * 60)
    for name in sorted(report.keys()):
        if name == "global":
            continue
        s = report[name]
        print(f"{name}:")
        print(f"Documents: {s['total_docs']:,}")
        print(f"Queries: {s['unique_queries']:,}")
        print(f"Docs/query: {s['docs_per_query']['mean']:.1f} (min={s['docs_per_query']['min']}, max={s['docs_per_query']['max']})")
        print(f"Features/doc: {s['features_per_doc']['mean']:.1f} (min={s['features_per_doc']['min']}, max={s['features_per_doc']['max']})")
        print(f"Sparsity: {s['avg_sparsity']*100:.1f}%")
        print()

    print(f"Full report saved to: {output_path}")


if __name__ == "__main__":
    main()
