import numpy as np
from .base import BaseArm


class GroundTruthArm(BaseArm):

    def __init__(self, name="ground_truth"):
        super().__init__(name)

    def train(self, train_records):
        self._is_trained = True

    def rank(self, record):
        indices = np.arange(record.num_items)
        neg_relevance = -record.relevance
        return indices[np.lexsort((indices, neg_relevance))]

    def compute_dcg(self, record, ranking, k=10):
        dcg = 0.0
        for rank, item_idx in enumerate(ranking[:k]):
            relevance = record.relevance[item_idx]
            dcg += relevance / np.log2(rank + 2)
        return dcg

    def compute_ndcg(self, record, ranking, k=10):
        ideal_dcg = self.compute_dcg(record, self.rank(record), k)
        if ideal_dcg == 0:
            return 1.0
        return self.compute_dcg(record, ranking, k) / ideal_dcg
