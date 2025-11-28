import os
import pickle
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA


def load_features_from_parquet(parquet_dir, max_samples=100000):
    parquet_path = Path(parquet_dir)
    all_features = []
    total_samples = 0

    for chunk_file in sorted(parquet_path.glob("*.parquet")):
        table = pq.read_table(chunk_file)

        for features_list in table["features"].to_pylist():
            all_features.extend(features_list)
            total_samples += len(features_list)

            if total_samples >= max_samples:
                break

        if total_samples >= max_samples:
            break

    features = np.array(all_features[:max_samples], dtype=np.float32)
    return features


def fit_and_save_pca(
    train_dir="data/processed/yahoo_parquet/set1/train",
    output_path="data/processed/pca_model.pkl",
    n_components=20,
    max_samples=100000,
    random_state=42,
):
    print(f"Loading training features from {train_dir}...")
    X = load_features_from_parquet(train_dir, max_samples)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")

    print(f"Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)

    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Explained variance: {explained_variance:.2%}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(pca, f)

    print(f"PCA model saved to {output_path}")

    return {
        "n_components": n_components,
        "n_samples_fitted": X.shape[0],
        "original_dim": X.shape[1],
        "explained_variance": float(explained_variance),
        "output_path": output_path,
    }


def load_pca(model_path="data/processed/pca_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"PCA model not found at {model_path}. "
            "Run fit_pca.py first to generate the model."
        )

    with open(model_path, "rb") as f:
        pca = pickle.load(f)

    return pca


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="data/processed/yahoo_parquet/set1/train")
    parser.add_argument("--output", default="data/processed/pca_model.pkl")
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=100000)

    args = parser.parse_args()

    result = fit_and_save_pca(
        train_dir=args.train_dir,
        output_path=args.output,
        n_components=args.n_components,
        max_samples=args.max_samples,
    )

    print("\nPCA fitting complete:")
    for key, val in result.items():
        print(f"{key}: {val}")
