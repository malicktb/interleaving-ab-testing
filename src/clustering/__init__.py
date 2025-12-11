"""Clustering for hierarchical ranker evaluation.

This module implements clustering protocols from main.tex:

OutputBasedClusterer (Section 3.1):
- Jaccard similarity on top-10 document IDs across 1000 queries
- HDBSCAN clustering with min_cluster_size=5
- Representative selection by highest offline NDCG@5

RandomClusterer (RQ3 Ablation):
- Random assignment to clusters (no behavioral similarity)
- First member as representative
- Proves clustering validity hypothesis
"""

from .output_based import OutputBasedClusterer, RandomClusterer, ClusteringResult

__all__ = ["OutputBasedClusterer", "RandomClusterer", "ClusteringResult"]
