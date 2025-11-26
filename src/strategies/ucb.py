"""
UCB Strategy (The Decision Maker).

This strategy implements the Multi-Dueling Bandit (MDB) algorithm described 
in the paper: "Multi-Dueling Bandits and Their Application to Online Ranker Evaluation" 
-- Brost, Seldin, Cox, & Lioma (2016)
"""

import numpy as np

from .base import BaseStrategy


class UCBSelectionStrategy(BaseStrategy):
    """UCB Selection Strategy (MDB Algorithm).

    Based on the logic defined in Brost et al. (2016), this strategy maintains
    two checklists (Confidence Bounds) to balance exploration and exploitation:

    1. Set E (Proven Winners): Arms that are statistically better than others.
       (Corresponds to the "Narrow Confidence Bound" in the paper).
    2. Set F (Potential Contenders): Arms that might be the best if we
       gave them a chance. (Corresponds to the "Wide Confidence Bound" in the paper).

    The Decision Rule (Algorithm 1 in the paper):
    - If there is exactly one Proven Winner, we stop testing and just use
      that one (Exploit).
    - If there are multiple (or zero) winners, we hold a tournament between
      everyone in the "Potential Contenders" list (Parallel Exploration).

    Attributes:
        alpha: Controls how optimistic we are (Higher = More exploration).
        beta: Controls how wide the "Contender" net is cast.
              See Equation (3) in the paper.
    """

    def __init__(self, arm_names, alpha=0.51, beta=1.0):
        """Initialize the UCB strategy.

        Args:
            arm_names: List of arm names to track.
            alpha: Exploration parameter (must be > 0.5). Controls optimism in UCB.
            beta: Parallelism parameter (must be >= 1.0). Widens the contender set.
        """
        super().__init__(arm_names)

        if alpha <= 0.5:
            raise ValueError("alpha must be > 0.5 to work correctly")
        if beta < 1.0:
            raise ValueError("beta must be >= 1.0")

        self.alpha = alpha
        self.beta = beta

    def _compute_ucb_matrix(self, beta_multiplier=1.0):
        """Calculate the Upper Confidence Bound for every pair of arms.

        This implements Equations (2) and (3) from the Brost et al. paper.

        Args:
            beta_multiplier: Scaling factor for confidence width (1.0 for Set E, beta for Set F).

        Returns:
            K x K matrix where cell [i, j] is the optimistic win rate of Arm i vs Arm j.
        """
        # Handle the very first step (log(0) is impossible)
        log_t = np.log(max(self.t, 1))

        # Find pairs we have actually tested
        explored = self.N > 0

        # Default: If we haven't tested a pair, assume the challenger wins 100% (Optimism)
        ucb = np.ones((self.K, self.K))

        if np.any(explored):
            # Current Win Rate (Empirical Estimate)
            empirical = np.divide(
                self.W, self.N, out=np.zeros_like(self.W), where=explored
            )
            # Uncertainty Bonus (The square root term in the paper's equations)
            exploration = np.sqrt(
                np.divide(
                    beta_multiplier * self.alpha * log_t,
                    self.N,
                    out=np.zeros_like(self.N),
                    where=explored,
                )
            )
            ucb = np.where(explored, empirical + exploration, 1.0)

        # Ignore self-comparisons (A vs A) by setting them to Infinity
        np.fill_diagonal(ucb, np.inf)

        return ucb

    def _compute_qualifying_set_vectorized(self, beta_multiplier):
        """Find arms that beat (or tie) EVERYONE else optimistically.

        An arm qualifies if its *lowest* predicted win rate against any opponent
        is still >= 50%.

        Args:
            beta_multiplier: Scaling factor for confidence width.

        Returns:
            Set of qualifying arm indices.
        """
        ucb_matrix = self._compute_ucb_matrix(beta_multiplier)

        # Find the minimum UCB value for each arm across all opponents
        min_ucb_per_arm = np.min(ucb_matrix, axis=1)
        
        # Who survives the worst matchup?
        qualifying = np.where(min_ucb_per_arm >= 0.5)[0]
        return set(qualifying.tolist())

    def compute_set_E(self):
        """Find the 'Proven Winners' using the Narrow Bound (Eq 2)."""
        return self._compute_qualifying_set_vectorized(beta_multiplier=1.0)

    def compute_set_F(self):
        """Find the 'Potential Contenders' using the Wide Bound (Eq 3)."""
        return self._compute_qualifying_set_vectorized(beta_multiplier=self.beta)

    def select_arms(self):
        """Decide who plays in the next round based on Set E and Set F.

        Returns:
            A set of arm names (e.g., {'Linear', 'Popularity'}).
        """
        E = self.compute_set_E()
        F = self.compute_set_F()

        if len(E) == 1:
            # Case 1: We found the single best arm. Use only it.
            selected_indices = E
        else:
            # Case 2: We are unsure. Let all potential contenders fight.
            # (If F is empty, fallback to testing everyone)
            selected_indices = F if F else set(range(self.K))

        return {self._idx_to_name(i) for i in selected_indices}

    def get_statistics(self):
        """Return debug info about the sets.

        Returns:
            Dictionary with W, N matrices, set_E, set_F, alpha, and beta.
        """
        stats = super().get_statistics()
        E = self.compute_set_E()
        F = self.compute_set_F()
        stats.update({
            "set_E": [self._idx_to_name(i) for i in E],
            "set_F": [self._idx_to_name(i) for i in F],
            "alpha": self.alpha,
            "beta": self.beta,
        })
        return stats
