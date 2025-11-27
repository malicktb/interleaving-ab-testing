from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
PARQUET_DIR = DATA_DIR / "parquet"


def verify_parquet_metadata(path):
    print(f"\nVerifying: {path.name}")
    print("-" * 40)

    pf = pq.ParquetFile(path)
    metadata = {
        "rows": pf.metadata.num_rows,
        "columns": pf.metadata.num_columns,
        "row_groups": pf.metadata.num_row_groups,
    }

    print(f"Rows: {metadata['rows']:,}")
    print(f"Columns: {metadata['columns']}")
    print(f"Row groups: {metadata['row_groups']}")

    return metadata


def verify_behaviors_schema(df):
    print("\n" + "=" * 60)
    print("BEHAVIORS SCHEMA")
    print("=" * 60)

    expected_columns = [
        "impression_id",
        "user_id",
        "timestamp",
        "history",
        "impression_news",
        "impression_labels",
    ]

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    missing = set(expected_columns) - set(df.columns)
    if missing:
        print(f"Missing columns: {missing}")
        return False

    for col in df.columns:
        print(f"{col}: {df[col].dtype}")

    return True


def verify_behaviors_consistency(df):
    print("\n" + "=" * 60)
    print("BEHAVIORS CONSISTENCY")
    print("=" * 60)

    news_lens = df["impression_news"].apply(len)
    label_lens = df["impression_labels"].apply(len)

    matching = (news_lens == label_lens).all()
    print(f"News/Labels length match: {matching}")

    all_labels = [label for labels in df["impression_labels"] for label in labels]
    unique_labels = set(all_labels)
    print(f"Unique label values: {unique_labels}")

    valid_labels = unique_labels <= {0, 1}
    print(f"Valid labels (0 or 1 only): {valid_labels}")

    print(f"Impressions per record: min={news_lens.min()}, max={news_lens.max()}, mean={news_lens.mean():.1f}")

    history_lens = df["history"].apply(len)
    print(f"History length: min={history_lens.min()}, max={history_lens.max()}, mean={history_lens.mean():.1f}")

    return matching and valid_labels


def verify_news_schema(df):
    print("\n" + "=" * 60)
    print("NEWS SCHEMA")
    print("=" * 60)

    expected_columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    missing = set(expected_columns) - set(df.columns)
    if missing:
        print(f"Missing columns: {missing}")
        return False

    for col in df.columns:
        print(f"{col}: {df[col].dtype}")

    return True


def verify_news_consistency(df):
    print("\n" + "=" * 60)
    print("NEWS CONSISTENCY")
    print("=" * 60)

    categories = df["category"].unique()
    print(f"Unique categories: {len(categories)}")
    print(f"Sample: {list(categories[:10])}")

    subcategories = df["subcategory"].unique()
    print(f"Unique subcategories: {len(subcategories)}")

    sample_entities = df["title_entities"].iloc[0] if len(df) > 0 else []
    print(f"Sample title_entities type: {type(sample_entities)}")
    if len(sample_entities) > 0:
        first_entity = sample_entities[0]
        print(f"First entity keys: {list(first_entity.keys()) if isinstance(first_entity, dict) else 'N/A'}")

    missing_titles = df["title"].isna().sum()
    missing_abstracts = (df["abstract"] == "").sum() + df["abstract"].isna().sum()
    print(f"Missing titles: {missing_titles}")
    print(f"Empty abstracts: {missing_abstracts}")

    return True


def verify_embeddings_schema(df, name):
    print(f"\n{'=' * 60}")
    print(f"{name.upper()} EMBEDDINGS SCHEMA")
    print("=" * 60)

    expected_columns = ["id", "embedding"]

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    missing = set(expected_columns) - set(df.columns)
    if missing:
        print(f"Missing columns: {missing}")
        return False

    if len(df) > 0:
        embedding_dims = df["embedding"].apply(len).unique()
        print(f"Embedding dimensions: {embedding_dims}")
        if len(embedding_dims) == 1:
            print(f"All embeddings have {embedding_dims[0]} dimensions")

    return True


def verify_labels(behaviors_df):
    print("\n" + "=" * 60)
    print("LABEL STATISTICS")
    print("=" * 60)

    all_labels = [label for labels in behaviors_df["impression_labels"] for label in labels]
    total = len(all_labels)
    positive = sum(all_labels)
    ctr = positive / total if total > 0 else 0

    print(f"Total impressions: {total:,}")
    print(f"Positive (clicked): {positive:,}")
    print(f"Negative (not clicked): {total - positive:,}")
    print(f"CTR: {ctr:.6f} ({ctr*100:.2f}%)")

    return {"total": total, "positive": positive, "ctr": ctr}


def run_all_validations(parquet_dir):
    print("=" * 60)
    print("MIND DATASET VALIDATION")
    print("=" * 60)
    print(f"Parquet directory: {parquet_dir}")

    results = {}

    behaviors_path = parquet_dir / "behaviors.parquet"
    if behaviors_path.exists():
        verify_parquet_metadata(behaviors_path)
        behaviors_df = pd.read_parquet(behaviors_path)
        results["behaviors_schema"] = verify_behaviors_schema(behaviors_df)
        results["behaviors_consistency"] = verify_behaviors_consistency(behaviors_df)
        verify_labels(behaviors_df)
    else:
        print(f"\nWarning: {behaviors_path} not found")

    news_path = parquet_dir / "news.parquet"
    if news_path.exists():
        verify_parquet_metadata(news_path)
        news_df = pd.read_parquet(news_path)
        results["news_schema"] = verify_news_schema(news_df)
        results["news_consistency"] = verify_news_consistency(news_df)
    else:
        print(f"\nWarning: {news_path} not found")

    entity_path = parquet_dir / "entity_embeddings.parquet"
    if entity_path.exists():
        verify_parquet_metadata(entity_path)
        entity_df = pd.read_parquet(entity_path)
        results["entity_embeddings"] = verify_embeddings_schema(entity_df, "entity")
    else:
        print(f"\nWarning: {entity_path} not found")

    relation_path = parquet_dir / "relation_embeddings.parquet"
    if relation_path.exists():
        verify_parquet_metadata(relation_path)
        relation_df = pd.read_parquet(relation_path)
        results["relation_embeddings"] = verify_embeddings_schema(relation_df, "relation")
    else:
        print(f"\nWarning: {relation_path} not found")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")

    all_passed = all(results.values()) if results else False
    print(f"\nOverall: {'PASSED' if all_passed else 'FAILED'}")

    return all_passed


def main():
    if not PARQUET_DIR.exists():
        print(f"Error: Parquet directory not found: {PARQUET_DIR}")
        print("Run convert_to_parquet.py first to generate parquet files.")
        return

    run_all_validations(PARQUET_DIR)


if __name__ == "__main__":
    main()
