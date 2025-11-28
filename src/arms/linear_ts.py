import numpy as np
from sklearn.decomposition import PCA
from .base import BaseArm


class LinearTSArm(BaseArm):

    def __init__(
        self,
        name="linear_ts",
        feature_dim=20,
        use_pca=True,
        regularization=1.0,
        noise_variance=1.0,
        seed=42,
    ):
        super().__init__(name)
        self.feature_dim = feature_dim
        self.use_pca = use_pca
        self.regularization = regularization
        self.noise_variance = noise_variance
        self.seed = seed
        self.B = None
        self.b = None
        self.pca = None
        self._B_inv = None
        self._B_inv_dirty = True
        self.rng = np.random.default_rng(seed)

    def _initialize_params(self, d):
        self.B = np.eye(d) * self.regularization
        self.b = np.zeros((d, 1))
        self._B_inv_dirty = True

    def _get_B_inv(self):
        if self._B_inv_dirty or self._B_inv is None:
            self._B_inv = np.linalg.solve(self.B, np.eye(self.B.shape[0]))
            self._B_inv_dirty = False
        return self._B_inv

    def _transform_features(self, features):
        if self.use_pca and self.pca is not None:
            return self.pca.transform(features)
        return features

    def train(self, train_records):
        if not train_records:
            print("Warning: No training data for LinearTS PCA fitting")
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

        if self.B is None:
            self._initialize_params(X.shape[1])

        B_inv = self._get_B_inv()
        mu = B_inv @ self.b

        try:
            L = np.linalg.cholesky(B_inv * self.noise_variance)
            z = self.rng.standard_normal((X.shape[1], 1))
            theta_sample = mu + L @ z
        except np.linalg.LinAlgError:
            eigenvals = np.linalg.eigvalsh(B_inv)
            min_eig = eigenvals.min()
            if min_eig < 0:
                B_inv_adjusted = B_inv - 1.1 * min_eig * np.eye(B_inv.shape[0])
                L = np.linalg.cholesky(B_inv_adjusted * self.noise_variance)
                z = self.rng.standard_normal((X.shape[1], 1))
                theta_sample = mu + L @ z
            else:
                theta_sample = mu

        scores = (X @ theta_sample).flatten()
        return np.argsort(scores)[::-1]

    def update(self, features, reward):
        if self.B is None:
            return

        if self.use_pca and self.pca is not None:
            x = self.pca.transform(features.reshape(1, -1)).flatten()
        else:
            x = features.flatten()

        x = x.reshape(-1, 1)
        self.B += (x @ x.T) / self.noise_variance
        self.b += (reward * x) / self.noise_variance
        self._B_inv_dirty = True

    def get_stats(self):
        stats = {
            "name": self.name,
            "feature_dim": self.feature_dim,
            "use_pca": self.use_pca,
            "noise_variance": self.noise_variance,
        }
        if self.B is not None:
            stats["trace_B"] = float(np.trace(self.B))
            stats["norm_b"] = float(np.linalg.norm(self.b))
        return stats
