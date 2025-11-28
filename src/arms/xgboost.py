import numpy as np
import xgboost as xgb
from .base import BaseArm


class XGBoostArm(BaseArm):

    def __init__(
        self,
        name="xgboost",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        train_fraction=1.0,
    ):
        super().__init__(name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.train_fraction = train_fraction
        self.model = None
        self.rng = np.random.default_rng(random_state)

    def train(self, train_records):
        if self.train_fraction < 1.0:
            n_samples = int(len(train_records) * self.train_fraction)
            indices = self.rng.choice(len(train_records), size=n_samples, replace=False)
            train_records = [train_records[i] for i in indices]

        X_list = []
        y_list = []
        group_sizes = []

        for record in train_records:
            X_list.append(record.features)
            y_list.append(record.relevance.astype(np.float32))
            group_sizes.append(record.num_items)

        if not X_list:
            print("Warning: No training data provided to XGBoostArm")
            self._is_trained = True
            return

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        self.model = xgb.XGBRanker(
            objective="rank:pairwise",
            tree_method="hist",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=0,
        )

        print(f"  Training XGBoost on {len(train_records)} queries, {len(y)} docs...")
        self.model.fit(X, y, group=group_sizes)
        self._is_trained = True

    def rank(self, record):
        if self.model is None:
            return np.arange(record.num_items)

        scores = self.model.predict(record.features)
        indices = np.arange(record.num_items)
        return indices[np.lexsort((indices, -scores))]
