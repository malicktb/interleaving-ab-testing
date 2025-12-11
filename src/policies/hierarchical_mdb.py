"""Hierarchical Multi-Dueling Bandit Policy with Knowledge Transfer.

Implements H-MDB-KT from main.tex Algorithm 1:
- Level 1: Cluster representatives (coarse screening)
- Level 2: Winning cluster members (fine-grained)

Key innovation: Knowledge Transfer (Warm Start) at level transition.
Per main.tex lines 74-76, cluster members inherit representative's
statistics instead of starting fresh (eliminates regret spike).
"""

import numpy as np
from typing import List, Optional, Dict

from .base import BasePolicy
from .trackers import StatisticsTrackerBase
from core.types import ClusteringResult


class HierarchicalMDBPolicy(BasePolicy):
    """H-MDB-KT: Two-level hierarchical MDB with Knowledge Transfer.

    Level 1: Evaluates cluster representatives (coarse screening)
    Level 2: Evaluates winning cluster members (fine-grained)

    Knowledge Transfer: At level transition, cluster members inherit
    representative's statistics (warm start) per main.tex Algorithm 1.
    """

    def __init__(
        self,
        arm_names: List[str],
        clustering_result: ClusteringResult,
        level1_rounds: int = 2000,
        alpha: float = 0.51,
        beta: float = 1.0,
        n_min: int = 0,
        warm_start: bool = True,
        statistics_tracker: Optional[StatisticsTrackerBase] = None,
    ):
        """Initialize hierarchical policy.

        Args:
            arm_names: All arm names (full pool).
            clustering_result: Result from OutputBasedClusterer or RandomClusterer.
            level1_rounds: Rounds before transition to Level 2.
            alpha: UCB exploration parameter (> 0.5).
            beta: Multiplier for Set F computation.
            n_min: Grace period before elimination.
            warm_start: If True, use Knowledge Transfer at level transition.
                       If False, reset statistics (ablation: no KT).
            statistics_tracker: Tracker for W/N (recommend DiscountedStatisticsTracker).
        """
        super().__init__(arm_names, statistics_tracker=statistics_tracker)

        if alpha <= 0.5:
            raise ValueError("alpha must be > 0.5")
        if level1_rounds < 1:
            raise ValueError("level1_rounds must be >= 1")

        self.clustering = clustering_result
        self.level1_rounds = level1_rounds
        self.alpha = alpha
        self.beta = beta
        self.n_min = n_min
        self.warm_start = warm_start

        # State machine
        self.current_level = 1
        self.winning_cluster_id: Optional[int] = None
        self._level1_active_arms: List[str] = []
        self._level2_active_arms: List[str] = []

        # Precompute Level 1 active arms (representatives only)
        self._level1_active_arms = self.clustering.get_level1_arms()

    def _get_active_arm_indices(self) -> List[int]:
        """Get indices of currently active arms."""
        if self.current_level == 1:
            active_names = self._level1_active_arms
        else:
            active_names = self._level2_active_arms
        return [self._name_to_idx(name) for name in active_names]

    def _compute_ucb_for_active(self, active_indices: List[int]) -> Dict[int, float]:
        """Compute UCB scores for active arms only (vectorized).

        Returns dict of arm_index -> min_UCB_score (minimum UCB across opponents).
        """
        if not active_indices:
            return {}

        log_t = np.log(max(self.t, 1))
        idx = np.array(active_indices, dtype=np.int32)
        n_active = len(idx)

        # Extract submatrices for active arms only: shape (n_active, n_active)
        W_sub = self.W[np.ix_(idx, idx)]
        N_sub = self.N[np.ix_(idx, idx)]

        # Compute UCB matrix for all pairs
        # For n_ij > 0: ucb = w_ij / n_ij + sqrt(alpha * log_t / n_ij)
        # For n_ij = 0: ucb = 1.0 (optimistic)
        with np.errstate(divide='ignore', invalid='ignore'):
            empirical = np.where(N_sub > 0, W_sub / N_sub, 0.0)
            exploration = np.where(N_sub > 0, np.sqrt(self.alpha * log_t / N_sub), 0.0)
            ucb_matrix = np.where(N_sub > 0, empirical + exploration, 1.0)

        # Set diagonal to inf (arm doesn't compete against itself)
        np.fill_diagonal(ucb_matrix, np.inf)

        # Min UCB for each arm is the minimum across its row (opponents)
        min_ucb = np.min(ucb_matrix, axis=1)

        return {active_indices[i]: float(min_ucb[i]) for i in range(n_active)}

    def _identify_winning_cluster(self) -> Optional[int]:
        """Identify winning cluster by highest row-sum of empirical wins.

        Per main.tex Algorithm 1 line 103: The winning cluster is determined
        by which representative has the highest sum of wins against all
        other Level 1 arms (pure empirical, no exploration bonus).

        This uses row-sum (sum of W[rep, j] for all active j != rep) rather
        than UCB scores to select the winning cluster deterministically
        based on accumulated empirical performance.
        """
        # Get indices of Level 1 active arms for row-sum computation
        active_indices = self._get_active_arm_indices()

        best_row_sum = -float('inf')
        best_cluster_id = None

        for cluster_id, rep_name in self.clustering.representatives.items():
            try:
                rep_idx = self._name_to_idx(rep_name)
            except ValueError:
                continue  # Representative not in arm_names

            # Sum wins against all other active arms (row-sum)
            row_sum = sum(self.W[rep_idx, j] for j in active_indices if j != rep_idx)

            if row_sum > best_row_sum:
                best_row_sum = row_sum
                best_cluster_id = cluster_id

        return best_cluster_id

    def _transition_to_level2(self):
        """Execute transition from Level 1 to Level 2 with Knowledge Transfer.

        Per main.tex Algorithm 1 lines 74-76:
            For k in C_best:
                N_k <- N_Rep(C_best); W_k <- W_Rep(C_best)

        Knowledge Transfer: Cluster members inherit representative's statistics.
        """
        # Identify winning cluster
        self.winning_cluster_id = self._identify_winning_cluster()

        if self.winning_cluster_id is None:
            # Fallback: use first cluster if no winner identified
            if self.clustering.clusters:
                self.winning_cluster_id = next(iter(self.clustering.clusters.keys()))
            else:
                # No clusters, keep all arms
                self._level2_active_arms = list(self.arm_names)
                self.current_level = 2
                return

        # Set Level 2 active arms (cluster members only)
        self._level2_active_arms = self.clustering.clusters.get(
            self.winning_cluster_id, []
        )

        # Knowledge Transfer (Warm Start) or Reset
        if self.warm_start:
            # Get representative for winning cluster
            rep_name = self.clustering.representatives.get(self.winning_cluster_id)
            if rep_name:
                rep_idx = self._name_to_idx(rep_name)
                # Get indices of cluster members (excluding representative)
                member_indices = [
                    self._name_to_idx(name)
                    for name in self._level2_active_arms
                    if name != rep_name
                ]
                # Inherit statistics from representative to all members
                if member_indices:
                    self._tracker.inherit_statistics(rep_idx, member_indices)
                    print(f"[H-MDB-KT] Knowledge Transfer: {rep_name} -> {len(member_indices)} members")
        else:
            # Ablation: Cold start (reset statistics)
            self._tracker.reset()
            print("[H-MDB] Cold Start: Statistics reset")

        # Update state
        self.current_level = 2

        print(f"[H-MDB] Transition to Level 2: winning_cluster={self.winning_cluster_id}")
        print(f"[H-MDB] Level 2 active arms: {self._level2_active_arms}")

    def select_arms(self) -> List[str]:
        """Select arms for current round."""
        # Check for level transition (happens AFTER level1_rounds)
        if self.current_level == 1 and self.t >= self.level1_rounds:
            self._transition_to_level2()

        # Get active arms for current level
        if self.current_level == 1:
            active_names = self._level1_active_arms
        else:
            active_names = self._level2_active_arms

        # Handle empty active set
        if not active_names:
            return list(self.arm_names)

        # During grace period, return all active arms
        if self.n_min > 0 and self.t < self.n_min:
            return active_names

        # Compute UCB-based selection (MDB algorithm)
        active_indices = [self._name_to_idx(name) for name in active_names]
        ucb_scores = self._compute_ucb_for_active(active_indices)

        # Set E: arms with min UCB >= 0.5
        set_E = [i for i, score in ucb_scores.items() if score >= 0.5]

        if len(set_E) == 1:
            # Exploit: single winner
            return [self._idx_to_name(set_E[0])]
        elif len(set_E) == 0:
            # All eliminated (shouldn't happen with alpha > 0.5)
            return active_names
        else:
            # Explore: all contenders
            return [self._idx_to_name(i) for i in set_E]

    def get_statistics(self) -> dict:
        """Get policy statistics for logging."""
        stats = super().get_statistics()
        stats.update({
            "current_level": self.current_level,
            "level1_rounds": self.level1_rounds,
            "winning_cluster_id": self.winning_cluster_id,
            "active_arms": (
                self._level1_active_arms if self.current_level == 1
                else self._level2_active_arms
            ),
            "n_clusters": len(self.clustering.clusters),
            "warm_start": self.warm_start,
            "alpha": self.alpha,
            "beta": self.beta,
            "n_min": self.n_min,
            "in_grace_period": self.n_min > 0 and self.t < self.n_min,
        })
        return stats
