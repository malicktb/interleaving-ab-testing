"""Output-based and random clustering for hierarchical ranker evaluation.

Implements clustering protocols from main.tex:

OutputBasedClusterer (Section 3.1):
- Jaccard similarity on top-10 doc IDs across 1000 queries
- HDBSCAN with min_cluster_size=5 to the similarity matrix
- Representative = highest offline NDCG@5 in cluster

RandomClusterer (RQ3 Ablation):
- Random assignment to clusters (no behavioral similarity)
- First member as representative
- Proves clustering validity hypothesis
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np

from core.base import BaseArm
from core.types import QueryRecord, ClusteringResult
from core.metrics import compute_ndcg


class OutputBasedClusterer:
    """Clusters static rankers by behavioral similarity (Jaccard on top-k outputs).

    Per main.tex Section 3.2:
    - Jaccard similarity on top-10 doc IDs across 1000 queries
    - HDBSCAN with min_cluster_size=5
    - Representative = highest offline NDCG@5 in cluster
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        k: int = 10,
        n_sample_queries: int = 1000,
        seed: int = 42,
    ):
        """Initialize the clusterer.

        Args:
            min_cluster_size: Minimum points for HDBSCAN cluster (default 5)
            k: Number of top documents for Jaccard similarity (default 10)
            n_sample_queries: Number of queries for similarity computation
            seed: Random seed for reproducibility
        """
        self.min_cluster_size = min_cluster_size
        self.k = k
        self.n_sample_queries = n_sample_queries
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def compute_jaccard_matrix(
        self,
        arms: Dict[str, BaseArm],
        sample_records: List[QueryRecord],
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute pairwise Jaccard similarity on top-k document sets.

        Jaccard(A, B) = |A intersection B| / |A union B|

        Args:
            arms: Dictionary of arm_name -> BaseArm (must be trained)
            sample_records: List of QueryRecord for similarity computation

        Returns:
            similarity_matrix: (n_arms, n_arms) symmetric matrix with values in [0, 1]
            arm_names: Ordered list of arm names corresponding to matrix indices
        """
        arm_names = list(arms.keys())
        n_arms = len(arm_names)
        n_queries = len(sample_records)

        if n_arms == 0:
            return np.array([[]]), arm_names

        if n_queries == 0:
            return np.eye(n_arms), arm_names

        # Collect top-k doc indices for each arm on each query
        # Shape: (n_arms, n_queries, k)
        effective_k = min(self.k, min(r.num_items for r in sample_records))
        top_k_docs = np.zeros((n_arms, n_queries, effective_k), dtype=np.int32)

        for i, arm_name in enumerate(arm_names):
            arm = arms[arm_name]
            for q, record in enumerate(sample_records):
                ranking = arm.rank(record)
                top_k_docs[i, q, :] = ranking[:effective_k]

        # Compute pairwise Jaccard similarity
        similarity = np.zeros((n_arms, n_arms), dtype=np.float64)

        for i in range(n_arms):
            similarity[i, i] = 1.0
            for j in range(i + 1, n_arms):
                total_jaccard = 0.0
                for q in range(n_queries):
                    set_i = set(top_k_docs[i, q, :])
                    set_j = set(top_k_docs[j, q, :])
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    if union > 0:
                        total_jaccard += intersection / union
                avg_jaccard = total_jaccard / n_queries
                similarity[i, j] = avg_jaccard
                similarity[j, i] = avg_jaccard

        return similarity, arm_names

    def cluster_arms(
        self,
        similarity_matrix: np.ndarray,
        arm_names: List[str],
    ) -> Tuple[np.ndarray, Dict[int, List[str]]]:
        """Apply HDBSCAN clustering to similarity matrix.

        HDBSCAN requires a distance matrix, so we convert:
        distance = 1 - similarity

        Args:
            similarity_matrix: (n_arms, n_arms) similarity values in [0, 1]
            arm_names: Ordered list of arm names

        Returns:
            labels: Cluster assignment for each arm (-1 = noise/singleton)
            clusters: Dictionary mapping cluster_id -> list of arm names
        """
        try:
            from hdbscan import HDBSCAN
        except ImportError:
            raise ImportError(
                "hdbscan package required. Install with: pip install hdbscan"
            )

        n_arms = len(arm_names)

        if n_arms < self.min_cluster_size:
            # Not enough arms to cluster, treat each as singleton
            labels = np.arange(n_arms)
            clusters = {i: [arm_names[i]] for i in range(n_arms)}
            return labels, clusters

        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix

        # Ensure diagonal is exactly 0 (numerical stability)
        np.fill_diagonal(distance_matrix, 0.0)

        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='precomputed',
        )
        labels = clusterer.fit_predict(distance_matrix)

        # Group arms by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(arm_names[idx])

        return labels, dict(clusters)

    def select_representatives(
        self,
        clusters: Dict[int, List[str]],
        arms: Dict[str, BaseArm],
        validation_records: List[QueryRecord],
        k: int = 5,
    ) -> Dict[int, str]:
        """Select representative for each cluster by highest offline NDCG@k.

        Args:
            clusters: Dictionary of cluster_id -> list of arm names
            arms: Dictionary of arm_name -> BaseArm
            validation_records: Records with relevance labels for NDCG
            k: Cutoff for NDCG computation (default 5)

        Returns:
            representatives: Dictionary of cluster_id -> representative arm name
        """
        representatives = {}

        for cluster_id, member_names in clusters.items():
            if cluster_id == -1:
                # Noise points will be handled separately
                continue

            if len(member_names) == 1:
                # Singleton cluster, arm is its own representative
                representatives[cluster_id] = member_names[0]
                continue

            best_arm = None
            best_ndcg = -1.0

            for arm_name in member_names:
                arm = arms[arm_name]
                total_ndcg = 0.0

                for record in validation_records:
                    ranking = arm.rank(record)
                    effective_k = min(k, len(ranking), len(record.relevance))
                    ranked_relevance = record.relevance[ranking[:effective_k]]
                    ndcg = compute_ndcg(ranked_relevance, record.relevance, k=effective_k)
                    total_ndcg += ndcg

                avg_ndcg = total_ndcg / len(validation_records) if validation_records else 0.0

                if avg_ndcg > best_ndcg:
                    best_ndcg = avg_ndcg
                    best_arm = arm_name

            if best_arm is not None:
                representatives[cluster_id] = best_arm

        return representatives

    def fit(
        self,
        arms: Dict[str, BaseArm],
        sample_records: List[QueryRecord],
        validation_records: List[QueryRecord],
    ) -> ClusteringResult:
        """Full clustering pipeline.

        Args:
            arms: Dictionary of arm_name -> BaseArm (must be trained)
            sample_records: Queries for Jaccard similarity (labels optional)
            validation_records: Queries with relevance for NDCG representative selection

        Returns:
            ClusteringResult with clusters and representatives
        """
        if not arms:
            # No arms to cluster
            return ClusteringResult(
                clusters={},
                representatives={},
                similarity_matrix=np.array([[]]),
                labels=np.array([]),
                arm_names=[],
            )

        # Compute similarity matrix
        similarity, arm_names = self.compute_jaccard_matrix(arms, sample_records)

        # Cluster
        labels, clusters = self.cluster_arms(similarity, arm_names)

        # Handle noise points (-1 label) as singleton clusters
        noise_arms = clusters.pop(-1, [])
        next_cluster_id = max(clusters.keys(), default=-1) + 1
        for arm_name in noise_arms:
            clusters[next_cluster_id] = [arm_name]
            next_cluster_id += 1

        # Select representatives
        representatives = self.select_representatives(
            clusters, arms, validation_records
        )

        return ClusteringResult(
            clusters=clusters,
            representatives=representatives,
            similarity_matrix=similarity,
            labels=labels,
            arm_names=arm_names,
        )


class RandomClusterer:
    """Random clustering baseline for RQ3 ablation.

    Groups arms randomly instead of by behavioral similarity. Per main.tex
    Table 1 Config 1.3 (Random-KT): proves that output-based clustering
    is necessary, not just having clusters.

    Key differences from OutputBasedClusterer:
    - No Jaccard similarity computation
    - Random arm assignment to clusters
    - First member of each cluster as representative
    """

    def __init__(
        self,
        n_clusters: int = 10,
        seed: int = 42,
    ):
        """Initialize random clusterer.

        Args:
            n_clusters: Number of clusters to create (default 10).
            seed: Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def fit(
        self,
        arms: Dict[str, Any],
        sample_records: List[QueryRecord] = None,
        validation_records: List[QueryRecord] = None,
    ) -> ClusteringResult:
        """Randomly assign arms to clusters.

        Args:
            arms: Dictionary of arm_name -> BaseArm.
            sample_records: Ignored (no similarity computation needed).
            validation_records: Ignored (first member is representative).

        Returns:
            ClusteringResult with random clusters.
        """
        arm_names = list(arms.keys())
        n_arms = len(arm_names)

        if n_arms == 0:
            return ClusteringResult(
                clusters={},
                representatives={},
                similarity_matrix=np.array([[]]),
                labels=np.array([]),
                arm_names=[],
            )

        # Shuffle arm names randomly
        shuffled_names = list(arm_names)
        self.rng.shuffle(shuffled_names)

        # Assign to clusters (round-robin for even distribution)
        clusters = defaultdict(list)
        labels = np.zeros(n_arms, dtype=np.int32)

        for idx, arm_name in enumerate(shuffled_names):
            cluster_id = idx % self.n_clusters
            clusters[cluster_id].append(arm_name)
            # Track label for original position (for compatibility)
            original_idx = arm_names.index(arm_name)
            labels[original_idx] = cluster_id

        # First member of each cluster is representative
        representatives = {}
        for cluster_id, members in clusters.items():
            representatives[cluster_id] = members[0]

        # Create fake similarity matrix (identity - no real similarity)
        similarity_matrix = np.eye(n_arms, dtype=np.float64)

        return ClusteringResult(
            clusters=dict(clusters),
            representatives=representatives,
            similarity_matrix=similarity_matrix,
            labels=labels,
            arm_names=arm_names,
        )
