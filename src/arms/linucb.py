import numpy as np
from sklearn.decomposition import PCA
from .base import BaseArm


class LinUCBArm(BaseArm):

    def __init__(self, name="linucb", alpha=1.0, feature_dim=20, use_pca=True, regularization=1.0):
        super().__init__(name)
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.use_pca = use_pca
        self.regularization = regularization
        self.A = None
        self.b = None
        self.pca = None
        self._A_inv = None
        self._A_inv_dirty = True

    def _initialize_params(self, d):
        self.A = np.eye(d) * self.regularization
        self.b = np.zeros((d, 1))
        self._A_inv_dirty = True

    def _get_A_inv(self):
        if self._A_inv_dirty or self._A_inv is None:
            self._A_inv = np.linalg.solve(self.A, np.eye(self.A.shape[0]))
            self._A_inv_dirty = False
        return self._A_inv

    def _transform_features(self, features):
        if self.use_pca and self.pca is not None:
            return self.pca.transform(features)
        return features

    def train(self, train_records):
        if not train_records:
            print("Warning: No training data for LinUCB PCA fitting")
            self._is_trained = True
            return

        raw_dim = train_records[0].features.shape[1]

        if self.use_pca and raw_dim > self.feature_dim:
            print(f"  Fitting PCA ({raw_dim} -> {self.feature_dim})...")

            all_features = []
            max_samples = 100000

            for record in train_records:
                all_features.append(record.features)
                if sum(f.shape[0] for f in all_features) >= max_samples:
                    break

            X = np.vstack(all_features)
            if len(X) > max_samples:
                X = X[:max_samples]

            self.pca = PCA(n_components=self.feature_dim, random_state=42)
            self.pca.fit(X)

            print(f"  PCA explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
            effective_dim = self.feature_dim
        else:
            self.use_pca = False
            effective_dim = raw_dim

        self._initialize_params(effective_dim)
        self._is_trained = True

    def rank(self, record):
        X = self._transform_features(record.features)

        if self.A is None:
            self._initialize_params(X.shape[1])

        A_inv = self._get_A_inv()
        theta = A_inv @ self.b

        means = (X @ theta).flatten()
        variances = np.sum((X @ A_inv) * X, axis=1)
        exploration = self.alpha * np.sqrt(np.maximum(0, variances))
        scores = means + exploration

        return np.argsort(scores)[::-1]

    def update(self, features, reward):
        if self.A is None:
            return

        if self.use_pca and self.pca is not None:
            x = self.pca.transform(features.reshape(1, -1)).flatten()
        else:
            x = features.flatten()

        x = x.reshape(-1, 1)
        self.A += x @ x.T
        self.b += reward * x
        self._A_inv_dirty = True

    def get_stats(self):
        stats = {
            "name": self.name,
            "alpha": self.alpha,
            "feature_dim": self.feature_dim,
            "use_pca": self.use_pca,
        }
        if self.A is not None:
            stats["trace_A"] = float(np.trace(self.A))
            stats["norm_b"] = float(np.linalg.norm(self.b))
        return stats
