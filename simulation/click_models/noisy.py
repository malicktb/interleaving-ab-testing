"""Noisy User Click Model.

Extends cascade model with realistic noise:
- Random clicks with probability noise_prob (uniform over positions)
- False negatives: relevant docs may be skipped with false_negative_rate
"""

from typing import List
import numpy as np

from core.base import ClickSimulator


class NoisyUserModel(ClickSimulator):
    """Cascade model with noise and false negatives.

    Behavior:
    1. With prob noise_prob: click a random position (ignoring relevance)
    2. Otherwise: scan cascade-style, but skip relevant docs with
       probability false_negative_rate (false negatives)

    Only one click per slate is possible.
    """

    def __init__(
        self,
        relevance_threshold: int = 3,
        noise_prob: float = 0.1,
        false_negative_rate: float = 0.1,
        max_depth: int = 10,
    ):
        """Initialize noisy user model.

        Args:
            relevance_threshold: Minimum relevance to trigger cascade click.
            noise_prob: Probability of random click (uniform over positions).
            false_negative_rate: Probability of skipping a relevant doc.
            max_depth: Maximum positions to scan.
        """
        self.relevance_threshold = relevance_threshold
        self.noise_prob = noise_prob
        self.false_negative_rate = false_negative_rate
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
            rng: Random number generator.

        Returns:
            List with single clicked position, or empty list.
        """
        effective_depth = min(len(slate), self.max_depth)
        if effective_depth == 0:
            return []

        if rng.random() < self.noise_prob:
            random_pos = rng.integers(0, effective_depth)
            return [random_pos]

        for pos, item_idx in enumerate(slate):
            if pos >= self.max_depth:
                break

            if item_idx < len(relevance):
                rel_grade = int(relevance[item_idx])

                if rel_grade >= self.relevance_threshold:
                    if rng.random() < self.false_negative_rate:
                        continue
                    return [pos]

        return []

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return (
            f"NoisyUserModel(threshold={self.relevance_threshold}, "
            f"noise={self.noise_prob:.2f}, fn_rate={self.false_negative_rate:.2f}, "
            f"max_depth={self.max_depth})"
        )

