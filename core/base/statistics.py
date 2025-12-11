"""Base class for statistics trackers.

Statistics trackers manage pairwise win (W) and comparison count (N) matrices
that bandit policies use to compute UCB bounds and select arms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np


class StatisticsTrackerBase(ABC):
    """Base interface for tracking pairwise W/N statistics.

    Tracks win (W) and comparison count (N) matrices that policies use
    to compute UCB/LCB bounds and select arms.

    Different implementations support:
    - Cumulative: All-time statistics
    - Sliding window: Recent rounds only
    - Discounted: Exponentially weighted statistics
    """

    def __init__(self, arm_names: List[str]):
        """Initialize tracker with ordered arm names.

        Args:
            arm_names: List of arm names in fixed order.
        """
        self.arm_names = arm_names
        # O(1) lookup map
        self.arm_index_map = {name: i for i, name in enumerate(arm_names)}
        self.K = len(arm_names)
        self.t = 0

    def _name_to_idx(self, name: str) -> int:
        """Convert arm name to index (O(1)).

        Args:
            name: Arm name.

        Returns:
            Index in the arm list.

        Raises:
            ValueError: If arm name not found.
        """
        try:
            return self.arm_index_map[name]
        except KeyError:
            raise ValueError(f"Arm '{name}' not found in tracker.")

    @abstractmethod
    def update(
        self,
        winner: Optional[str],
        participants: List[str],
        pairwise_outcomes: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> None:
        """Update statistics with round results.

        Args:
            winner: Name of winning arm (for legacy single-winner updates).
            participants: List of arm names that participated this round.
            pairwise_outcomes: Optional dict of (winner, loser) -> credit.
                If provided, used instead of legacy winner/participants.
        """
        pass

    @abstractmethod
    def get_W(self) -> np.ndarray:
        """Return current win matrix W[i,j] = wins of i over j."""
        pass

    @abstractmethod
    def get_N(self) -> np.ndarray:
        """Return current comparison count matrix N[i,j]."""
        pass

    def get_effective_t(self) -> int:
        """Return effective time horizon for UCB calculation.

        For cumulative tracker, this is total rounds.
        For windowed tracker, this may be min(t, window_size).
        """
        return self.t

    def get_win_rate(self, i: int, j: int) -> float:
        """Get empirical win rate of arm i vs arm j.

        Args:
            i: Index of first arm.
            j: Index of second arm.

        Returns:
            Win rate in [0, 1], or 0.5 if no comparisons.
        """
        N = self.get_N()
        W = self.get_W()
        if N[i, j] == 0:
            return 0.5  # Prior for unexplored pairs
        return W[i, j] / N[i, j]

    def get_statistics(self) -> Dict:
        """Get current statistics for logging/debugging.

        Returns:
            Dict with t, effective_t, W, and N matrices.
        """
        return {
            "t": self.t,
            "effective_t": self.get_effective_t(),
            "W": self.get_W().tolist(),
            "N": self.get_N().tolist(),
        }

    def reset(self) -> None:
        """Reset statistics to initial state.

        Override in subclasses to clear W/N matrices. Used for
        hierarchical policies that reset between levels.

        Default implementation only resets round counter.
        """
        self.t = 0
