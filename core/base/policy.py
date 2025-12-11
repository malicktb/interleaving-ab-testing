"""Base class for bandit policies.

Policies select which arms to evaluate each round and update
based on feedback from multileaving attribution.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
import numpy as np

from core.base.statistics import StatisticsTrackerBase


class BasePolicy(ABC):
    """Base class for bandit policies.

    Supports optional statistics tracker injection for flexible W/N management.
    If no tracker is provided, uses cumulative statistics (backward compatible).

    Subclasses must implement:
    - select_arms(): Choose arms for the current round
    """

    def __init__(
        self,
        arm_names: List[str],
        statistics_tracker: Optional[StatisticsTrackerBase] = None,
    ):
        """Initialize policy.

        Args:
            arm_names: List of arm names.
            statistics_tracker: Optional tracker for W/N management.
                If None, a CumulativeStatisticsTracker will be created.
        """
        self.arm_names = arm_names
        self.K = len(arm_names)
        # O(1) name lookup map
        self._arm_index_map = {name: i for i, name in enumerate(arm_names)}

        # Store tracker reference - will be set by subclass or during setup
        self._tracker = statistics_tracker

    def _name_to_idx(self, name: str) -> int:
        """Convert arm name to index (O(1) lookup).

        Args:
            name: Arm name.

        Returns:
            Index in arm list.

        Raises:
            ValueError: If arm not found.
        """
        try:
            return self._arm_index_map[name]
        except KeyError:
            raise ValueError(f"Arm '{name}' not found in policy.")

    def _idx_to_name(self, idx: int) -> str:
        """Convert index to arm name.

        Args:
            idx: Index in arm list.

        Returns:
            Arm name.
        """
        return self.arm_names[idx]

    @property
    def t(self) -> int:
        """Current round number (delegated to tracker)."""
        return self._tracker.t if self._tracker else 0

    @property
    def W(self) -> np.ndarray:
        """Win matrix (delegated to tracker)."""
        return self._tracker.get_W() if self._tracker else np.zeros((self.K, self.K))

    @property
    def N(self) -> np.ndarray:
        """Comparison count matrix (delegated to tracker)."""
        return self._tracker.get_N() if self._tracker else np.zeros((self.K, self.K))

    @abstractmethod
    def select_arms(self) -> List[str]:
        """Select arms to evaluate this round.

        Returns:
            List of arm names to include in multileaving.
        """
        pass

    def update(
        self,
        winner: Optional[str],
        participants: List[str],
        pairwise_outcomes: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> None:
        """Update policy with round results.

        Args:
            winner: Name of winning arm, or None if no click.
            participants: List of arm names that participated.
            pairwise_outcomes: Optional explicit pairwise outcomes for W/N.
        """
        if self._tracker:
            self._tracker.update(
                winner=winner,
                participants=participants,
                pairwise_outcomes=pairwise_outcomes,
            )

    def get_win_rate(self, i: int, j: int) -> float:
        """Get empirical win rate of arm i vs arm j.

        Args:
            i: Index of first arm.
            j: Index of second arm.

        Returns:
            Win rate in [0, 1].
        """
        if self._tracker:
            return self._tracker.get_win_rate(i, j)
        return 0.5

    def get_statistics(self) -> dict:
        """Get policy statistics for logging.

        Returns:
            Dict with current statistics.
        """
        if self._tracker:
            return self._tracker.get_statistics()
        return {"t": 0}
