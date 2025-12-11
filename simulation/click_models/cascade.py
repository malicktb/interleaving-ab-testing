"""Cascade Click Model.

Models user as scanning top-down and clicking the first sufficiently
relevant document. User stops after the first click (or at max_depth).
"""

from typing import List
import numpy as np

from core.base import ClickSimulator


class CascadeModel(ClickSimulator):
    """Cascade model: user clicks first relevant doc, then stops.

    The user scans positions 0, 1, 2, ... and clicks the first document
    with relevance >= threshold. Only one click per slate is possible.
    """

    def __init__(self, relevance_threshold: int = 3, max_depth: int = 10):
        """Initialize cascade model.

        Args:
            relevance_threshold: Minimum relevance grade to trigger click.
            max_depth: Maximum positions to scan before giving up.
        """
        self.relevance_threshold = relevance_threshold
        self.max_depth = max_depth

    def simulate(
        self,
        slate: List[int],
        relevance: np.ndarray,
        rng: np.random.Generator,
    ) -> List[int]:
        """Simulate clicks on a slate.

        Args:
            slate: Document indices in display order.
            relevance: Relevance grades for all documents.
            rng: Random number generator (not used in deterministic cascade).

        Returns:
            List with single clicked position, or empty list.
        """
        for pos, item_idx in enumerate(slate):
            if pos >= self.max_depth:
                break

            if item_idx < len(relevance):
                rel_grade = int(relevance[item_idx])
                if rel_grade >= self.relevance_threshold:
                    return [pos]

        return []

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return (
            f"CascadeModel(threshold={self.relevance_threshold}, "
            f"max_depth={self.max_depth})"
        )

