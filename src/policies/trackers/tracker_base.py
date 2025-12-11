"""Base class for statistics trackers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np


class StatisticsTrackerBase(ABC):
    """Base interface for tracking pairwise W/N statistics.

    Tracks win (W) and comparison count (N) matrices that policies use
    to compute UCB/LCB bounds and select arms.
    """

    def __init__(self, arm_names: List[str]):
        """Initialize tracker with ordered arm names."""
        self.arm_names = arm_names
        # OPTIMIZATION: O(1) lookup map
        self.arm_index_map = {name: i for i, name in enumerate(arm_names)}
        self.K = len(arm_names)
        self.t = 0

    def _name_to_idx(self, name: str) -> int:
        """Convert arm name to index (O(1))."""
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

    def get_statistics(self) -> Dict:
        """Get current statistics for logging/debugging."""
        # Note: calling .tolist() on large matrices can be slow for logging.
        # Ensure this is only called periodically or at end of episode.
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

    @abstractmethod
    def inherit_statistics(
        self,
        from_arm_idx: int,
        to_arm_indices: List[int],
    ) -> None:
        """Copy W/N statistics from source arm to target arms (warm start).

        Used for Knowledge Transfer at hierarchical level transitions.
        Algorithm 1 lines 74-76:
            For k in C_best:
                N_k <- N_Rep(C_best); W_k <- W_Rep(C_best)

        This copies the representative's statistics to all cluster members,
        eliminating the "cold start" regret spike.

        Args:
            from_arm_idx: Index of source arm (cluster representative).
            to_arm_indices: Indices of target arms (cluster members).
        """
        pass
