"""Discounted statistics tracker for non-stationary arms.

Implements exponential discounting:
- W^disc uses γ decay (recent wins weighted more)
- N uses undiscounted counts (for UCB confidence width)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from .tracker_base import StatisticsTrackerBase


class DiscountedStatisticsTracker(StatisticsTrackerBase):
    """Exponential discounting tracker for non-stationary environments.

    Key difference from cumulative tracker:
    - W matrix is discounted by γ each round (recent wins matter more)
    - N matrix remains undiscounted (for proper UCB confidence bounds)

    This follows the conservative discounting approach from Garivier & Moulines (2011)
    where estimation uses discounted statistics but exploration uses full counts.

    Formula:
        W^disc_t = γ * W^disc_{t-1} + ΔW_t
        N_ij = undiscounted symmetric counts

        u_ij = w^disc_ij / (w^disc_ij + w^disc_ji) + √(α ln t / N_ij)
    """

    def __init__(
        self,
        arm_names: List[str],
        discount_factor: float = 0.995,
    ):
        """Initialize discounted tracker.

        Args:
            arm_names: List of arm names in order.
            discount_factor: Decay factor γ ∈ (0, 1). Default 0.995 gives
                effective horizon of ~200 rounds (1/(1-γ)).
        """
        super().__init__(arm_names)

        if not 0 < discount_factor < 1:
            raise ValueError(
                f"discount_factor must be in (0, 1), got {discount_factor}"
            )

        self.gamma = discount_factor

        # Discounted win matrix - decays each round
        self.W_disc = np.zeros((self.K, self.K), dtype=np.float64)

        # Undiscounted comparison counts - for UCB confidence
        self.N = np.zeros((self.K, self.K), dtype=np.float64)

    def update(
        self,
        winner: Optional[str],
        participants: List[str],
        pairwise_outcomes: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> None:
        """Update discounted statistics.

        Args:
            winner: Name of winning arm.
            participants: Arms that participated.
            pairwise_outcomes: Optional explicit pairwise outcomes.
        """
        self.t += 1

        # Decay all existing wins by γ (O(K²) but K is typically small)
        self.W_disc *= self.gamma

        if pairwise_outcomes is not None:
            # Use explicit pairwise outcomes
            for (winner_name, loser_name), credit in pairwise_outcomes.items():
                winner_idx = self._name_to_idx(winner_name)
                loser_idx = self._name_to_idx(loser_name)
                # Add new win (unweighted - will be discounted in future rounds)
                self.W_disc[winner_idx, loser_idx] += credit
                # Undiscounted comparison counts
                self.N[winner_idx, loser_idx] += 1
                self.N[loser_idx, winner_idx] += 1
        elif winner is not None:
            # Legacy single-winner update
            winner_idx = self._name_to_idx(winner)
            for name in participants:
                if name != winner:
                    loser_idx = self._name_to_idx(name)
                    self.W_disc[winner_idx, loser_idx] += 1
                    self.N[winner_idx, loser_idx] += 1
                    self.N[loser_idx, winner_idx] += 1

    def get_W(self) -> np.ndarray:
        """Return discounted win matrix."""
        return self.W_disc

    def get_N(self) -> np.ndarray:
        """Return undiscounted comparison count matrix."""
        return self.N

    def reset(self) -> None:
        """Reset all statistics (for hierarchical level transitions).

        Clears W, N matrices and resets round counter to 0.
        """
        self.W_disc.fill(0)
        self.N.fill(0)
        self.t = 0

    def inherit_statistics(
        self,
        from_arm_idx: int,
        to_arm_indices: list,
    ) -> None:
        """Copy W/N statistics from source arm to target arms (warm start).

        Knowledge Transfer: copies the representative's pairwise statistics
        to cluster members, giving them a "warm start" at Level 2.

        For discounted tracker, copies both W_disc (discounted wins) and
        N (undiscounted counts) matrices.

        Args:
            from_arm_idx: Index of source arm (cluster representative).
            to_arm_indices: Indices of target arms (cluster members).
        """
        for to_idx in to_arm_indices:
            if to_idx == from_arm_idx:
                continue  # Skip self-copy

            # Copy row and column from source to target
            # W_disc: discounted wins
            self.W_disc[to_idx, :] = self.W_disc[from_arm_idx, :].copy()
            self.W_disc[:, to_idx] = self.W_disc[:, from_arm_idx].copy()
            # N: undiscounted comparison counts
            self.N[to_idx, :] = self.N[from_arm_idx, :].copy()
            self.N[:, to_idx] = self.N[:, from_arm_idx].copy()

    def get_statistics(self) -> Dict:
        """Get statistics with tracker type info."""
        stats = super().get_statistics()
        stats["tracker_type"] = "discounted"
        stats["discount_factor"] = self.gamma
        stats["effective_horizon"] = 1.0 / (1.0 - self.gamma)
        return stats
