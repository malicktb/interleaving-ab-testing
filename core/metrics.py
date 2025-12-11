"""Core metrics for ranking evaluation.

This module contains fundamental metrics used throughout the codebase:
- DCG/NDCG computation
- RegretTelemetry for tracking simulation quality

For specialized tracking (SuccessRateTracker, WildcardSurvivalTracker),
see analysis.tracking module.
"""

from collections import defaultdict
from typing import List, Dict, Any, Optional
import numpy as np


# Cache for DCG discount factors
_discount_cache: Dict[int, np.ndarray] = {}


def _get_discounts(k: int) -> np.ndarray:
    """Return cached DCG discount factors 1/log2(rank+1) for ranks [1..k].

    Args:
        k: Number of positions.

    Returns:
        Array of discount factors.
    """
    if k not in _discount_cache:
        positions = np.arange(1, k + 1, dtype=np.float32)
        _discount_cache[k] = 1.0 / np.log2(positions + 1)
    return _discount_cache[k]


def compute_dcg(relevance: np.ndarray, k: int = 10) -> float:
    """Compute Discounted Cumulative Gain.

    DCG = sum(relevance[i] / log2(i + 2)) for i in [0, k)

    Args:
        relevance: Relevance scores in ranking order.
        k: Cutoff position (default 10).

    Returns:
        DCG score.
    """
    relevance = np.asarray(relevance, dtype=np.float32)[:k]

    if relevance.size == 0:
        return 0.0

    discounts = _get_discounts(len(relevance))
    return float(np.dot(relevance, discounts))


def compute_ndcg(
    relevance: np.ndarray,
    ideal_relevance: Optional[np.ndarray] = None,
    k: int = 10
) -> float:
    """Compute Normalized Discounted Cumulative Gain.

    NDCG = DCG / IDCG where IDCG is the ideal (maximum possible) DCG.

    Args:
        relevance: Relevance scores in ranking order (position 0 = top result).
        ideal_relevance: Optional ideal relevance ordering. If None, the ideal
            is computed by sorting `relevance` in descending order (standard
            NDCG definition: ideal = best possible reordering of same docs).
        k: Cutoff position for DCG computation.

    Returns:
        NDCG score in [0, 1]. Returns 1.0 if ideal_dcg is 0 (no relevant docs).
    """
    dcg = compute_dcg(relevance, k)

    if ideal_relevance is None:
        # Standard NDCG: ideal is the best possible ordering of the same documents
        # (sort relevance descending to get maximum achievable DCG)
        ideal_for_dcg = np.sort(np.asarray(relevance))[::-1]
    else:
        ideal_for_dcg = ideal_relevance

    ideal_dcg = compute_dcg(ideal_for_dcg, k)

    if ideal_dcg == 0:
        # No relevant documents - NDCG is defined as 1.0 (trivially optimal)
        return 1.0

    return dcg / ideal_dcg


class RegretTelemetry:
    """Track regret and selection statistics across simulation iterations.

    Records per-iteration:
    - Which arms were selected
    - Which arm won (got clicked)
    - NDCG regret (gap from optimal ranking)

    Provides summary statistics for analysis.
    """

    def __init__(self, arm_names: List[str]):
        """Initialize regret telemetry.

        Args:
            arm_names: List of all arm names.
        """
        self.arm_names = arm_names
        self._init_state()

    def _init_state(self) -> None:
        """Initialize or reset all mutable state."""
        self.num_iterations = 0

        self.selection_history: List[List[str]] = []
        self.winner_history: List[Optional[str]] = []
        self.click_history: List[int] = []
        self.ndcg_regret_history: List[float] = []
        self.cumulative_ndcg_regret_history: List[float] = []

        self.total_ndcg_regret = 0.0
        self.arm_selection_counts: Dict[str, int] = defaultdict(int)
        self.arm_win_counts: Dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        """Reset all telemetry to initial state.

        Clears all histories and counters. Safe to call between
        experiments or episodes for reuse.
        """
        self._init_state()

    def record_iteration(
        self,
        selected_arms: List[str],
        winner: Optional[str]
    ) -> None:
        """Record an iteration's results.

        Args:
            selected_arms: Arms that participated this iteration.
            winner: Arm that won (got clicked), or None.
        """
        self.num_iterations += 1

        self.selection_history.append(selected_arms)
        for arm in selected_arms:
            self.arm_selection_counts[arm] += 1

        self.winner_history.append(winner)
        has_click = winner is not None
        self.click_history.append(1 if has_click else 0)

        if has_click:
            self.arm_win_counts[winner] += 1

    def record_ndcg_regret(self, optimal_ndcg: float, shown_ndcg: float) -> None:
        """Record NDCG regret for an iteration.

        Regret = optimal_ndcg - shown_ndcg (clamped to >= 0).

        Args:
            optimal_ndcg: NDCG of the optimal (oracle) ranking.
            shown_ndcg: NDCG of the ranking shown to user.
        """
        ndcg_regret = optimal_ndcg - shown_ndcg
        ndcg_regret = max(0.0, ndcg_regret)

        self.ndcg_regret_history.append(ndcg_regret)
        self.total_ndcg_regret += ndcg_regret
        self.cumulative_ndcg_regret_history.append(self.total_ndcg_regret)

    def get_arm_selection_rates(self) -> Dict[str, float]:
        """Get selection rate for each arm.

        Returns:
            Dict of arm_name -> selection rate in [0, 1].
        """
        if self.num_iterations == 0:
            return {}
        return {
            arm: count / self.num_iterations
            for arm, count in self.arm_selection_counts.items()
        }

    def get_arm_win_rates(self) -> Dict[str, float]:
        """Get win rate for each arm (proportion of clicks).

        Returns:
            Dict of arm_name -> win rate in [0, 1].
        """
        total_clicks = sum(self.click_history)
        if total_clicks == 0:
            return {}
        return {
            arm: count / total_clicks
            for arm, count in self.arm_win_counts.items()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dict with iterations, regret stats, and click stats.
        """
        return {
            "iterations": self.num_iterations,
            "total_ndcg_regret": self.total_ndcg_regret,
            "avg_ndcg_regret": (
                self.total_ndcg_regret / self.num_iterations
                if self.num_iterations > 0 else 0.0
            ),
            "total_clicks": sum(self.click_history),
            "click_rate": (
                sum(self.click_history) / self.num_iterations
                if self.num_iterations > 0 else 0.0
            ),
        }
