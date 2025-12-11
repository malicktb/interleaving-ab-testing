from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
import numpy as np

from .trackers import StatisticsTrackerBase, CumulativeStatisticsTracker


class BasePolicy(ABC):
    """Base class for bandit policies.

    Supports optional statistics tracker injection for flexible W/N management.
    If no tracker is provided, uses cumulative statistics (backward compatible).
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
                If None, uses CumulativeStatisticsTracker (default behavior).
        """
        self.arm_names = arm_names
        self.K = len(arm_names)
        # O(1) name lookup map
        self._arm_index_map = {name: i for i, name in enumerate(arm_names)}

        # Use provided tracker or create default cumulative tracker
        self._tracker = statistics_tracker or CumulativeStatisticsTracker(arm_names)

    def _name_to_idx(self, name: str) -> int:
        """Convert arm name to index (O(1) lookup)."""
        try:
            return self._arm_index_map[name]
        except KeyError:
            raise ValueError(f"Arm '{name}' not found in policy.")

    def _idx_to_name(self, idx: int) -> str:
        return self.arm_names[idx]

    @property
    def t(self) -> int:
        """Current round number (delegated to tracker)."""
        return self._tracker.t

    @property
    def W(self) -> np.ndarray:
        """Win matrix (delegated to tracker)."""
        return self._tracker.get_W()

    @property
    def N(self) -> np.ndarray:
        """Comparison count matrix (delegated to tracker)."""
        return self._tracker.get_N()

    @abstractmethod
    def select_arms(self) -> List[str]:
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
        # Accept optional pairwise outcomes for algorithms that produce multiple winners
        # (e.g., PPM/SOSM attribution). Default to legacy single-winner update.
        self._tracker.update(
            winner=winner,
            participants=participants,
            pairwise_outcomes=pairwise_outcomes,
        )

    def get_statistics(self) -> dict:
        """Get policy statistics for logging."""
        return self._tracker.get_statistics()
