import numpy as np
from .base import BaseArm


class SingleFeatureArm(BaseArm):

    def __init__(self, name="single_feature", feature_idx=0, descending=True):
        super().__init__(name)
        self.feature_idx = feature_idx
        self.descending = descending

    def train(self, train_records):
        self._is_trained = True

    def rank(self, record):
        feature_values = record.features[:, self.feature_idx]
        indices = np.arange(record.num_items)

        if self.descending:
            return indices[np.lexsort((indices, -feature_values))]
        else:
            return indices[np.lexsort((indices, feature_values))]
