"""Shared data types used across the codebase.

This module contains dataclasses that are used by multiple packages:
- QueryRecord: Query with features and relevance labels
- AttributionResult: Result of credit attribution
- ClusteringResult: Result of arm clustering
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class QueryRecord:
    """A single query with document features and relevance labels.

    Attributes:
        query_id: Unique identifier for the query.
        num_items: Number of documents for this query.
        features: Document feature matrix (num_items x feature_dim).
        relevance: Relevance labels for each document (num_items,).
    """
    query_id: str
    num_items: int
    features: np.ndarray
    relevance: np.ndarray

    def __repr__(self) -> str:
        """Concise representation for debugging (avoids printing full arrays)."""
        feat_shape = self.features.shape if self.features is not None else None
        return (
            f"QueryRecord(query_id={self.query_id!r}, "
            f"num_items={self.num_items}, "
            f"features_shape={feat_shape})"
        )


@dataclass
class AttributionResult:
    """Structured result from attribution computation.

    Attributes:
        pairwise_outcomes: Dict mapping (winner_arm, loser_arm) -> credit weight.
            Used by policies to update W/N matrices.
        winner: Single winner arm name for backward compatibility with
            legacy code that expects a single winner per round.
        credits: Per-arm credit scores (useful for debugging/analysis).
    """
    pairwise_outcomes: Dict[Tuple[str, str], float] = field(default_factory=dict)
    winner: Optional[str] = None
    credits: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClusteringResult:
    """Result of output-based clustering.

    Used by hierarchical policies to determine arm groupings.

    Attributes:
        clusters: Mapping of cluster_id -> list of arm names in cluster.
        representatives: Mapping of cluster_id -> representative arm name.
        similarity_matrix: Pairwise Jaccard similarity (n_arms x n_arms).
        labels: HDBSCAN cluster labels for each arm.
        arm_names: Ordered list of arm names corresponding to matrix rows.
    """
    clusters: Dict[int, List[str]]
    representatives: Dict[int, str]
    similarity_matrix: np.ndarray
    labels: np.ndarray
    arm_names: List[str]

    def get_level1_arms(self) -> List[str]:
        """Get arms active in Level 1: representatives.

        Returns:
            List of arm names for Level 1 evaluation.
        """
        return list(self.representatives.values())

