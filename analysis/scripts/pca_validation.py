import pickle
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.fit_pca import load_features_from_parquet


def analyze_pca_variance(
    train_dir="data/processed/yahoo_parquet/set1/train",
    pca_path="data/processed/pca_model.pkl",
    max_samples=50000,
):
    print("Loading training features...")
    X = load_features_from_parquet(train_dir, max_samples)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")

    print("Loading PCA model...")
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    print(f"PCA components: {pca.n_components_}")

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    total_variance = pca.explained_variance_ratio_.sum()

    return {
        "n_components": pca.n_components_,
        "total_variance": float(total_variance),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumulative_variance.tolist(),
        "components_80": int(np.searchsorted(cumulative_variance, 0.80) + 1),
        "components_90": int(np.searchsorted(cumulative_variance, 0.90) + 1),
        "components_95": int(np.searchsorted(cumulative_variance, 0.95) + 1),
    }


def print_results(results):
    print("\n" + "=" * 60)
    print("PCA Variance Analysis")
    print("=" * 60)

    print(f"\nTotal components: {results['n_components']}")
    print(f"Total explained variance: {results['total_variance']:.2%}")
    print(f"Components needed for 80%: {results['components_80']}")
    print(f"Components needed for 90%: {results['components_90']}")
    print(f"Components needed for 95%: {results['components_95']}")

    print("\nPer-component variance (top 10):")
    for i, var in enumerate(results["explained_variance_ratio"][:10]):
        cumulative = results["cumulative_variance"][i]
        print(f"PC{i+1}: {var:.4f} ({cumulative:.2%} cumulative)")


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("PCA Variance Analysis\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total components: {results['n_components']}\n")
        f.write(f"Total explained variance: {results['total_variance']:.2%}\n")
        f.write(f"Components needed for 80%: {results['components_80']}\n")
        f.write(f"Components needed for 90%: {results['components_90']}\n")
        f.write(f"Components needed for 95%: {results['components_95']}\n")

        f.write("\nPer-component variance:\n")
        for i, var in enumerate(results["explained_variance_ratio"]):
            cumulative = results["cumulative_variance"][i]
            f.write(f"PC{i+1}: {var:.4f} ({cumulative:.2%} cumulative)\n")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="data/processed/yahoo_parquet/set1/train")
    parser.add_argument("--pca-path", default="data/processed/pca_model.pkl")
    parser.add_argument("--output", default="analysis/pca_validation.txt")

    args = parser.parse_args()

    results = analyze_pca_variance(args.train_dir, args.pca_path)
    print_results(results)
    save_results(results, args.output)
