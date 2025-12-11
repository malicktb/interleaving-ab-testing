"""MultiRUCB policy per Du et al. (2021) specification."""

import numpy as np
from typing import List, Optional

from .base import BasePolicy
from .trackers import StatisticsTrackerBase


class MultiRUCBPolicy(BasePolicy):
    """MultiRUCB policy per Du et al. (2021) specification from main.tex.

    Computes pairwise UCB bounds:
        u_ij = w_ij / (w_ij + w_ji) + sqrt(alpha * ln(t) / (w_ij + w_ji))
        Convention: x/0 := 1 for any x
        u_ii = 1/2 for all i

    Candidate set:
        C = { x_c | u_cj >= 1/2 for all j in {1,...,K} }

    Selection logic based on |C|:
        - C empty: sample m arms uniformly from all arms
        - |C| = 1: set B <- C, return C
        - 1 < |C| <= m: return all of C
        - |C| > m: sample m from C with preference to B

    Maintains hypothesized best arm B (singleton set).

    Args:
        arm_names: List of arm names.
        alpha: UCB exploration parameter (must be > 0.5).
        m: Comparison set size. If None, defaults to K (all arms).
        n_min: Grace period rounds before selection logic activates.
        statistics_tracker: Optional tracker for W/N management.
        seed: Random seed for reproducible sampling.
    """

    def __init__(
        self,
        arm_names: List[str],
        alpha: float = 0.51,
        m: int = None,
        n_min: int = 0,
        statistics_tracker: Optional[StatisticsTrackerBase] = None,
        seed: int = None,
    ):
        super().__init__(arm_names, statistics_tracker=statistics_tracker)

        if alpha <= 0.5:
            raise ValueError("alpha must be > 0.5")
        if m is not None and m < 1:
            raise ValueError("m must be >= 1")

        self.alpha = alpha
        self.m = m if m is not None else self.K
        self.n_min = n_min
        self.B: Optional[int] = None  # Hypothesized best arm index (singleton set)
        self._rng = np.random.default_rng(seed)

    def _compute_ucb_matrix(self) -> np.ndarray:
        """Compute pairwise UCB matrix per Du et al. specification.

        Formula:
            u_ij = w_ij / (w_ij + w_ji) + sqrt(alpha * ln(t) / (w_ij + w_ji))
            Convention: x/0 := 1 for any x
            u_ii = 1/2 for all i

        Returns:
            UCB matrix of shape (K, K).
        """
        W = self.W

        # Compute total pairwise comparisons: w_ij + w_ji
        # This is symmetric: total[i,j] = total[j,i]
        total = W + W.T

        t = max(self.t, 1)
        log_t = np.log(t)

        # Mask for pairs with at least one comparison
        explored = total > 0

        # Compute empirical win probability: w_ij / (w_ij + w_ji)
        # Convention: x/0 := 1 (initialize to 1.0 for unexplored)
        empirical = np.divide(
            W, total,
            out=np.ones((self.K, self.K), dtype=np.float64),
            where=explored
        )

        # Compute exploration bonus: sqrt(alpha * ln(t) / (w_ij + w_ji))
        # Convention: sqrt(x/0) := 1 (for unexplored pairs, UCB = 1)
        exploration = np.sqrt(
            np.divide(
                self.alpha * log_t,
                total,
                out=np.ones((self.K, self.K), dtype=np.float64),
                where=explored
            )
        )

        # UCB = empirical + exploration for explored pairs
        # UCB = 1 (from x/0 := 1 convention) for unexplored pairs
        ucb = np.where(explored, empirical + exploration, 1.0)

        # Set diagonal: u_ii = 1/2 for all i
        np.fill_diagonal(ucb, 0.5)

        return ucb

    def _compute_candidate_set(self, ucb_matrix: np.ndarray) -> np.ndarray:
        """Compute candidate set C: arms whose UCB >= 0.5 against all others.

        C = { x_c | u_cj >= 1/2, for all j in {1,...,K} }

        Args:
            ucb_matrix: Pairwise UCB matrix.

        Returns:
            Array of arm indices in candidate set C.
        """
        # Check if each arm's UCB against ALL other arms is >= 0.5
        # min over row gives worst-case UCB for that arm
        min_ucb_per_arm = np.min(ucb_matrix, axis=1)

        # Arms qualifying for C: min UCB >= 0.5
        candidate_mask = min_ucb_per_arm >= 0.5

        return np.where(candidate_mask)[0]

    def _sample_from_candidates(self, C: np.ndarray) -> List[str]:
        """Sample m arms from candidate set C with preference to B.

        If B is empty or not in C: sample m uniformly from C
        If B is set and in C:
            - With prob 1/2: include B, fill m-1 from C minus B
            - With prob 1/2: sample m uniformly from C minus B

        Args:
            C: Array of candidate arm indices (|C| > m guaranteed).

        Returns:
            List of m selected arm names.
        """
        C_set = set(C.tolist())

        # If B is not set or B is not in C, sample uniformly from C
        if self.B is None or self.B not in C_set:
            selected = self._rng.choice(C, size=self.m, replace=False)
            return [self._idx_to_name(i) for i in sorted(selected)]

        # B is set and B in C
        C_minus_B = np.array([x for x in C if x != self.B])

        # With probability 1/2, include B
        if self._rng.random() < 0.5:
            # Include B, sample m-1 from C \ B
            others = self._rng.choice(C_minus_B, size=self.m - 1, replace=False)
            selected = np.concatenate([[self.B], others])
        else:
            # Sample m uniformly from C \ B
            # Note: C \ B has |C| - 1 elements, and |C| > m, so |C \ B| >= m
            selected = self._rng.choice(C_minus_B, size=self.m, replace=False)

        return [self._idx_to_name(i) for i in sorted(selected)]

    def select_arms(self) -> List[str]:
        """Select arms using 4-case MultiRUCB logic per Du et al.

        Selection logic:
        1. C empty: sample m arms uniformly from all arms
        2. |C| = 1: set B <- C, return C
        3. 1 < |C| <= m: return all of C
        4. |C| > m: sample m from C with preference to B

        Returns:
            List of selected arm names.
        """
        # Grace period: return all arms
        if self.n_min > 0 and self.t < self.n_min:
            return list(self.arm_names)

        # Compute UCB matrix and candidate set
        ucb_matrix = self._compute_ucb_matrix()
        C = self._compute_candidate_set(ucb_matrix)

        # Case 1: C is empty - rare fallback
        if len(C) == 0:
            # Sample m arms uniformly from full set
            all_arms = np.arange(self.K)
            selected = self._rng.choice(all_arms, size=min(self.m, self.K), replace=False)
            return [self._idx_to_name(i) for i in sorted(selected)]

        # Case 2: |C| = 1
        if len(C) == 1:
            # Update B to be the single candidate
            self.B = int(C[0])
            return [self._idx_to_name(C[0])]

        # Case 3: 1 < |C| <= m
        if len(C) <= self.m:
            # Return all candidates
            return [self._idx_to_name(i) for i in sorted(C)]

        # Case 4: |C| > m - need to sample
        return self._sample_from_candidates(C)

    def get_statistics(self) -> dict:
        """Get policy statistics for logging/debugging."""
        stats = super().get_statistics()

        ucb_matrix = self._compute_ucb_matrix()
        C = self._compute_candidate_set(ucb_matrix)

        stats.update({
            "policy_type": "multi_rucb",
            "alpha": self.alpha,
            "m": self.m,
            "n_min": self.n_min,
            "in_grace_period": self.n_min > 0 and self.t < self.n_min,
            "candidate_set_C": [self._idx_to_name(i) for i in C],
            "candidate_set_size": len(C),
            "hypothesized_best_B": self._idx_to_name(self.B) if self.B is not None else None,
        })
        return stats
