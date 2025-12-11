"""Cumulative statistics tracker (all-time W/N totals)."""

from typing import Dict, List, Tuple, Optional
import numpy as np

from .tracker_base import StatisticsTrackerBase


class CumulativeStatisticsTracker(StatisticsTrackerBase):
    """All-time W/N tracker; every round contributes equally forever."""

    def __init__(self, arm_names: List[str]):
        """Initialize cumulative tracker.

        Args:
            arm_names: List of arm names in order.
        """
        super().__init__(arm_names)

        # Cumulative matrices - never reset
        self.W = np.zeros((self.K, self.K), dtype=np.float32)
        self.N = np.zeros((self.K, self.K), dtype=np.float32)

    def update(
        self,
        winner: Optional[str],
        participants: List[str],
        pairwise_outcomes: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> None:
        """Update cumulative statistics.

        Args:
            winner: Name of winning arm.
            participants: Arms that participated.
            pairwise_outcomes: Optional explicit pairwise outcomes.
        """
        self.t += 1

        if pairwise_outcomes is not None:
            # Use explicit pairwise outcomes
            for (winner_name, loser_name), credit in pairwise_outcomes.items():
                winner_idx = self._name_to_idx(winner_name)
                loser_idx = self._name_to_idx(loser_name)
                self.W[winner_idx, loser_idx] += credit
                self.N[winner_idx, loser_idx] += 1
                self.N[loser_idx, winner_idx] += 1
        elif winner is not None:
            # Legacy single-winner update
            winner_idx = self._name_to_idx(winner)
            for name in participants:
                if name != winner:
                    loser_idx = self._name_to_idx(name)
                    self.W[winner_idx, loser_idx] += 1
                    self.N[winner_idx, loser_idx] += 1
                    self.N[loser_idx, winner_idx] += 1

    def get_W(self) -> np.ndarray:
        """Return cumulative win matrix."""
        return self.W

    def get_N(self) -> np.ndarray:
        """Return cumulative comparison count matrix."""
        return self.N

    def reset(self) -> None:
        """Reset cumulative statistics to initial state.

        Clears W and N matrices and resets round counter.
        Used for hierarchical policies that reset between levels.
        """
        super().reset()  # Reset t = 0
        self.W.fill(0)
        self.N.fill(0)

    def inherit_statistics(
        self,
        from_arm_idx: int,
        to_arm_indices: list,
    ) -> None:
        """Copy W/N statistics from source arm to target arms (warm start).

        Knowledge Transfer: copies the representative's pairwise statistics
        to cluster members, giving them a "warm start" at Level 2.

        Args:
            from_arm_idx: Index of source arm (cluster representative).
            to_arm_indices: Indices of target arms (cluster members).
        """
        for to_idx in to_arm_indices:
            if to_idx == from_arm_idx:
                continue  # Skip self-copy

            # Copy row and column from source to target
            # This preserves the representative's comparison history
            self.W[to_idx, :] = self.W[from_arm_idx, :].copy()
            self.W[:, to_idx] = self.W[:, from_arm_idx].copy()
            self.N[to_idx, :] = self.N[from_arm_idx, :].copy()
            self.N[:, to_idx] = self.N[:, from_arm_idx].copy()

    def get_statistics(self) -> Dict:
        """Get statistics with tracker type info."""
        stats = super().get_statistics()
        stats["tracker_type"] = "cumulative"
        return stats
