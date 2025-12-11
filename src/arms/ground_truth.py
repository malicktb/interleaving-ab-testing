import numpy as np
from .base import BaseArm
from core.metrics import compute_dcg, compute_ndcg


class GroundTruthArm(BaseArm):

    def __init__(self, name="ground_truth"):
        super().__init__(name)

    def train(self, train_records):
        self._is_trained = True

    def rank(self, record):
        indices = np.arange(record.num_items)
        neg_relevance = -record.relevance
        return indices[np.lexsort((indices, neg_relevance))]

    def get_ndcg(self, record, ranking, k=10):
        """Compute NDCG for a given ranking using centralized metrics.

        Args:
            record: QueryRecord with relevance labels.
            ranking: List of document indices in ranked order.
            k: Cutoff for NDCG computation.

        Returns:
            NDCG@k score.
        """
        # Get relevance scores in the order of the ranking
        relevance = np.array([record.relevance[idx] for idx in ranking[:k]])
        # Ideal relevance is sorted descending
        ideal_relevance = np.sort(record.relevance)[::-1][:k]
        return compute_ndcg(relevance, ideal_relevance, k)
