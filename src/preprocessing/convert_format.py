from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_letor_line(line):
    parts = line.strip().split()
    if len(parts) < 2:
        return None

    relevance = int(parts[0])
    query_id = parts[1].split(":")[1]

    features = {}
    for part in parts[2:]:
        if part.startswith("#"):
            break
        if ":" in part:
            fid, val = part.split(":")
            features[int(fid)] = float(val)

    return relevance, query_id, features


def densify_features(features_dict, num_features=700):
    dense = np.zeros(num_features, dtype=np.float32)
    for fid, val in features_dict.items():
        if 1 <= fid <= num_features:
            dense[fid - 1] = val
    return dense


def group_by_query(lines, num_features=700):
    current_qid = None
    features_list = []
    relevance_list = []

    for line in lines:
        parsed = parse_letor_line(line)
        if parsed is None:
            continue

        rel, qid, feat_dict = parsed

        if qid != current_qid and current_qid is not None:
            yield {
                "query_id": current_qid,
                "features": np.array(features_list, dtype=np.float32),
                "relevance": np.array(relevance_list, dtype=np.int8),
            }
            features_list = []
            relevance_list = []

        current_qid = qid
        features_list.append(densify_features(feat_dict, num_features))
        relevance_list.append(rel)

    if current_qid is not None and features_list:
        yield {
            "query_id": current_qid,
            "features": np.array(features_list, dtype=np.float32),
            "relevance": np.array(relevance_list, dtype=np.int8),
        }


def convert_file(input_path, output_dir, chunk_size=10000, num_features=700):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        ("query_id", pa.string()),
        ("features", pa.list_(pa.list_(pa.float32()))),
        ("relevance", pa.list_(pa.int8())),
    ])

    with open(input_path, "r") as f:
        lines = f.readlines()

    queries = list(group_by_query(lines, num_features))
    total_queries = len(queries)

    chunk_idx = 0
    for i in range(0, total_queries, chunk_size):
        chunk = queries[i:i + chunk_size]

        arrays = {
            "query_id": [q["query_id"] for q in chunk],
            "features": [q["features"].tolist() for q in chunk],
            "relevance": [q["relevance"].tolist() for q in chunk],
        }

        table = pa.table(arrays, schema=schema)
        output_path = Path(output_dir) / f"chunk_{chunk_idx:04d}.parquet"
        pq.write_table(table, output_path, compression="snappy")
        chunk_idx += 1

    return total_queries


def convert_letor_to_parquet(
    raw_dir="data/raw/yahoo_dataset",
    output_dir="data/processed/yahoo_parquet",
    num_features=700,
    chunk_size=10000,
):
    raw_path = Path(raw_dir)
    results = {}

    for txt_file in raw_path.glob("*.txt"):
        name = txt_file.stem
        parts = name.split(".")

        if len(parts) >= 2:
            set_name = parts[0]
            split = parts[1]
        else:
            set_name = "set1"
            split = name

        split_output = Path(output_dir) / set_name / split
        print(f"Converting {txt_file.name} -> {split_output}/")

        count = convert_file(txt_file, split_output, chunk_size, num_features)
        results[f"{set_name}/{split}"] = count
        print(f"{count} queries converted")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/yahoo_dataset")
    parser.add_argument("--output-dir", default="data/processed/yahoo_parquet")
    parser.add_argument("--num-features", type=int, default=700)
    parser.add_argument("--chunk-size", type=int, default=10000)

    args = parser.parse_args()

    results = convert_letor_to_parquet(
        args.raw_dir, args.output_dir, args.num_features, args.chunk_size
    )

    print("\nConversion complete:")
    for split, count in results.items():
        print(f"{split}: {count} queries")
