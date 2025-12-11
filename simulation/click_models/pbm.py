"""Position-Based Click Model (PBM).

Models user clicks as: P(click) = P(examine | position) * P(click | relevance).
Examination probability decreases with position, allowing multiple clicks.
"""

from typing import List, Dict, Optional
import numpy as np

from core.base import ClickSimulator


class PositionBasedModel(ClickSimulator):
    """Position-Based Model for click simulation.

    In PBM, clicks are independent at each position:
        P(click at pos) = P(examine | pos) * P(click | relevance)

    This allows multiple clicks per slate (unlike cascade models).
    """

    DEFAULT_EXAMINATION_PROBS = [
        1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10
    ]

    DEFAULT_CLICK_PROBS = {
        0: 0.01,
        1: 0.10,
        2: 0.30,
        3: 0.60,
        4: 0.95,
    }

    def __init__(
        self,
        examination_probs: Optional[List[float]] = None,
        click_probs: Optional[Dict[int, float]] = None,
        max_positions: int = 10,
    ):
        """Initialize PBM.

        Args:
            examination_probs: P(examine | position) for each position.
            click_probs: P(click | relevance) mapping relevance -> probability.
            max_positions: Maximum positions to consider.
        """
        # Use 'is not None' to correctly handle empty list []
        if examination_probs is not None:
            self.examination_probs = list(examination_probs)
        else:
            self.examination_probs = self.DEFAULT_EXAMINATION_PROBS.copy()

        self.click_probs = click_probs or self.DEFAULT_CLICK_PROBS.copy()
        self.max_positions = max_positions

        # Extend examination_probs to max_positions if needed
        if not self.examination_probs:
            # Empty list provided - use default decay starting from 1.0
            self.examination_probs = [1.0]
        while len(self.examination_probs) < max_positions:
            last = self.examination_probs[-1]
            self.examination_probs.append(max(0.01, last * 0.9))

        # Precompute numpy arrays for fast simulation
        self._examination_probs = np.asarray(self.examination_probs, dtype=np.float32)
        max_rel = max(self.click_probs.keys()) if self.click_probs else 0
        self._click_prob_lookup = np.zeros(max_rel + 1, dtype=np.float32)
        for rel, prob in self.click_probs.items():
            if rel >= 0:
                if rel >= len(self._click_prob_lookup):
                    new_lookup = np.zeros(rel + 1, dtype=np.float32)
                    new_lookup[: len(self._click_prob_lookup)] = self._click_prob_lookup
                    self._click_prob_lookup = new_lookup
                self._click_prob_lookup[rel] = prob

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
            List of clicked positions.
        """
        if not slate:
            return []

        max_len = min(len(slate), self.max_positions, len(self._examination_probs))
        if max_len == 0:
            return []

        slate_arr = np.asarray(slate[:max_len], dtype=np.int64)

        # Guard against slate indices that exceed relevance length
        valid_mask = slate_arr < len(relevance)
        rel_grades = np.zeros(max_len, dtype=np.int64)
        rel_grades[valid_mask] = relevance[slate_arr[valid_mask]].astype(np.int64)

        rel_grades = np.minimum(rel_grades, len(self._click_prob_lookup) - 1)

        p_click_given_rel = self._click_prob_lookup[rel_grades]
        p_examine = self._examination_probs[:max_len]
        click_probs = p_examine * p_click_given_rel

        random_draws = rng.random(max_len)
        clicked_positions = np.nonzero(random_draws < click_probs)[0]

        return clicked_positions.tolist()

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        p_top = self.examination_probs[0] if self.examination_probs else 0.0
        p_rel4 = self.click_probs.get(4, 0.0)
        return (
            f"PositionBasedModel(max_pos={self.max_positions}, "
            f"p_examine_top={p_top:.2f}, p_click_rel4={p_rel4:.2f})"
        )

