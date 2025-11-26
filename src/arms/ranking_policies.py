"""
Deterministic ranking policies (arms).

These are the "Smart" arms that use machine learning or statistics to 
generate rankings, as opposed to the Random arm.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from .base import BaseArm


class LinearArm(BaseArm):
    """Ranking policy using Logistic Regression.

    This arm learns a weight for every feature (e.g., Price, Category, User Segment)
    to predict how likely a user is to click an item.

    It combines 4 types of features into a single 27-dimensional vector:
    1. Basic Dense Features (7)
    2. Optional Dense Features (12)
    3. User Context Features (3)
    4. Item Category Features (5)
    """

    def __init__(self, name="linear", max_iter=1000, random_state=42):
        """Initialize the linear arm.

        Args:
            name: Unique identifier for this arm.
            max_iter: Maximum iterations for logistic regression convergence.
            random_state: Seed for reproducibility.
        """
        super().__init__(name)
        self.scaler = StandardScaler()
        self.model = None
        self.max_iter = max_iter
        self.random_state = random_state

    def extract_features(self, record):
        """Convert a single data record into a feature matrix.

        Args:
            record: Dictionary containing item and user features.

        Returns:
            A (num_items, num_features) numpy array.
        """
        # 1. Stack the item-specific dense features
        d1 = np.vstack(record["items_dense_features"])
        d2 = np.vstack(record["items_dense_features2"])

        # 2. Repeat user features for each item (so every item gets the context)
        num_items = len(record["items_dense_features"])
        user_prof = np.tile(record["context_features"], (num_items, 1))

        # 3. Stack category IDs (treated as numbers)
        cats = np.vstack(record["items_features"]).astype(np.float64)

        # 4. Combine everything side-by-side
        return np.hstack([d1, d2, user_prof, cats])

    def train(self, sample_df):
        """Train the model using a sample of historical data.

        Args:
            sample_df: DataFrame containing training records with features and labels.
        """
        X_list = []
        y_list = []

        # Convert dataframe rows into feature matrices
        for _, record in sample_df.iterrows():
            X = self.extract_features(record)
            y = np.array(record["labels"])

            X_list.append(X)
            y_list.append(y)

        # Stack into one large training set
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)

        # Remove any rows with missing/NaN values to prevent crashing
        valid_mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]

        # Standardize features (scale to mean=0, var=1)
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model (Auto-tunes regularization and balances classes)
        self.model = LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 7),
            cv=3,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.model.fit(X_train_scaled, y_train)
        self._is_trained = True

    def rank(self, record):
        """Rank items by predicted click probability.

        Args:
            record: Dictionary containing item features.

        Returns:
            Array of item indices sorted by predicted click probability (highest first).
        """
        X = self.extract_features(record)

        # Fill missing values with 0
        X = np.nan_to_num(X, nan=0.0)

        # Scale features using stats learned during training
        X_scaled = self.scaler.transform(X)

        # Predict probability of "Click" (Class 1)
        probs = self.model.predict_proba(X_scaled)[:, 1]

        # Sort indices: Highest probability first
        return np.argsort(probs)[::-1]


class PopularityArm(BaseArm):
    """Ranking policy based on Historical Popularity (CTR).

    Since items don't have simple IDs, this arm hashes the item's features
    into "Buckets." It then tracks how often items in that bucket get clicked.
    """

    def __init__(self, name="popularity", num_buckets=10000, alpha=1.0, beta=1.0, seed=42):
        """Initialize the popularity arm.

        Args:
            name: Unique identifier for this arm.
            num_buckets: Number of hash buckets for item grouping.
            alpha: Dirichlet prior smoothing for clicks.
            beta: Dirichlet prior smoothing for non-clicks.
            seed: Seed for random projection weights.
        """
        super().__init__(name)
        self.num_buckets = num_buckets
        self.alpha = alpha  # Smoothing for Clicks
        self.beta = beta    # Smoothing for Non-Clicks
        self.ctr_table = defaultdict(lambda: [0, 0])

        # Generate random weights once (used for hashing)
        rng = np.random.default_rng(seed)
        self.proj_dense = rng.integers(1, 1000, size=7)
        self.proj_cats = rng.integers(1, 1000, size=5)

    def _compute_hashes(self, dense_features, item_features):
        """Map items to bucket IDs using random projections.

        Args:
            dense_features: Array of dense item features.
            item_features: Array of categorical item features.

        Returns:
            Array of bucket IDs for each item.
        """
        d_quant = (dense_features * 100).astype(int)
        c_int = item_features.astype(int)

        # Weighted sum of features -> Single integer
        h_dense = np.dot(d_quant, self.proj_dense)
        h_cats = np.dot(c_int, self.proj_cats)

        return (h_dense + h_cats) % self.num_buckets

    def _process_chunk(self, chunk_df):
        """Update popularity stats using a chunk of data.

        Args:
            chunk_df: DataFrame chunk containing items and labels.
        """
        # Flatten the data (batch process all items in the chunk)
        batch_dense = np.vstack([np.vstack(row) for row in chunk_df["items_dense_features"].values])
        batch_cats = np.vstack([np.vstack(row) for row in chunk_df["items_features"].values])
        batch_labels = np.concatenate(chunk_df["labels"].values)

        buckets = self._compute_hashes(batch_dense, batch_cats)

        # Update counts
        for bucket, label in zip(buckets, batch_labels):
            self.ctr_table[bucket][1] += 1  # Count Impression
            if label == 1.0:
                self.ctr_table[bucket][0] += 1  # Count Click

    def train(self, train_data=None):
        """Placeholder. This arm uses streaming training via _process_chunk."""
        raise NotImplementedError(
            "PopularityArm uses streaming training. Call _process_chunk() instead."
        )

    def _get_ctr(self, bucket):
        """Calculate smoothed CTR using Dirichlet prior.

        Args:
            bucket: The bucket ID to look up.

        Returns:
            Smoothed CTR value: (Clicks + alpha) / (Impressions + alpha + beta).
        """
        clicks, impressions = self.ctr_table[bucket]
        return (clicks + self.alpha) / (impressions + self.alpha + self.beta)

    def rank(self, record):
        """Rank items by their bucket's popularity.

        Args:
            record: Dictionary containing item features.

        Returns:
            Array of item indices sorted by CTR (highest first).
        """
        d_feats = np.vstack(record["items_dense_features"])
        c_feats = np.vstack(record["items_features"])

        buckets = self._compute_hashes(d_feats, c_feats)
        scores = np.array([self._get_ctr(b) for b in buckets])

        # Sort indices: Highest CTR first
        return np.argsort(scores)[::-1]