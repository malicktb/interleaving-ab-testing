import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = DATA_DIR / "parquet"


def parse_impressions(s):
    if pd.isna(s) or not s:
        return [], []
    news, labels = [], []
    for item in s.split():
        parts = item.rsplit("-", 1)
        if len(parts) == 2:
            news.append(parts[0])
            labels.append(int(parts[1]))
    return news, labels


def convert_behaviors():
    input_path = DATA_DIR / "behaviors.tsv"
    output_path = OUTPUT_DIR / "behaviors.parquet"

    print(f"behaviors: Starting ({input_path.stat().st_size / 1e9:.2f} GB)", flush=True)

    schema = pa.schema([
        ("impression_id", pa.string()),
        ("user_id", pa.string()),
        ("timestamp", pa.string()),
        ("history", pa.list_(pa.string())),
        ("impression_news", pa.list_(pa.string())),
        ("impression_labels", pa.list_(pa.int64())),
    ])

    writer = pq.ParquetWriter(output_path, schema, compression="snappy")
    total = 0

    for i, chunk in enumerate(pd.read_csv(
        input_path, sep="\t", header=None,
        names=["impression_id", "user_id", "timestamp", "history", "impressions"],
        dtype=str, chunksize=100_000
    )):
        chunk["history"] = chunk["history"].apply(lambda x: x.split() if pd.notna(x) and x else [])
        parsed = chunk["impressions"].apply(parse_impressions)
        chunk["impression_news"] = parsed.apply(lambda x: x[0])
        chunk["impression_labels"] = parsed.apply(lambda x: x[1])
        chunk = chunk.drop(columns=["impressions"])

        table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
        writer.write_table(table)

        total += len(chunk)
        print(f"behaviors: {total:,} records written", flush=True)

    writer.close()
    print(f"behaviors: Done ({total:,} records)", flush=True)
    return total


def convert_news():
    input_path = DATA_DIR / "news.tsv"
    output_path = OUTPUT_DIR / "news.parquet"

    print(f"news: Starting ({input_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    df = pd.read_csv(
        input_path, sep="\t", header=None,
        names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"],
        dtype=str
    )

    def parse_entities(s):
        if pd.isna(s) or not s or s == "[]":
            return []
        try:
            return json.loads(s)
        except:
            return []

    df["abstract"] = df["abstract"].fillna("")
    df["title_entities"] = df["title_entities"].apply(parse_entities)
    df["abstract_entities"] = df["abstract_entities"].apply(parse_entities)

    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), output_path, compression="snappy")

    print(f"news: Done ({len(df):,} records)", flush=True)
    return len(df)


def convert_embeddings(name, filename):
    input_path = DATA_DIR / filename
    output_path = OUTPUT_DIR / f"{name}.parquet"

    print(f"{name}: Starting ({input_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    schema = pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.float64()))])
    writer = pq.ParquetWriter(output_path, schema, compression="snappy")

    ids, embeddings = [], []
    total = 0

    with open(input_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                ids.append(parts[0])
                embeddings.append([float(x) for x in parts[1].split()])
            else:
                parts = line.strip().split()
                if len(parts) > 1:
                    ids.append(parts[0])
                    embeddings.append([float(x) for x in parts[1:]])

            if len(ids) >= 10_000:
                writer.write_table(pa.table({"id": ids, "embedding": embeddings}, schema=schema))
                total += len(ids)
                ids, embeddings = [], []

    if ids:
        writer.write_table(pa.table({"id": ids, "embedding": embeddings}, schema=schema))
        total += len(ids)

    writer.close()
    print(f"{name}: Done ({total:,} records)", flush=True)
    return total


def main():
    print("=" * 50, flush=True)
    print("MIND Dataset -> Parquet Conversion", flush=True)
    print("=" * 50, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()
    stats = {}

    tasks = [
        ("behaviors", convert_behaviors),
        ("news", convert_news),
        ("entity_embeddings", lambda: convert_embeddings("entity_embeddings", "entity_embedding.vec")),
        ("relation_embeddings", lambda: convert_embeddings("relation_embeddings", "relation_embedding.vec")),
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fn): name for name, fn in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                stats[name] = future.result()
            except Exception as e:
                print(f"{name}: FAILED - {e}", flush=True)

    print("=" * 50, flush=True)
    print("Summary", flush=True)
    print("=" * 50, flush=True)
    for name, count in stats.items():
        print(f"{name}: {count:,} records", flush=True)
    print(f"Total time: {time.time() - start:.1f}s", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
