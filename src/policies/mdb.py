import numpy as np
from typing import List, Optional
from .base import BasePolicy
from .trackers import StatisticsTrackerBase


class MDBPolicy(BasePolicy):
    """Multi-Dueling Bandit policy with UCB-based arm selection.

    Uses Set E (likely winners) and Set F (contenders) to select arms.
    Supports optional statistics tracker for windowed statistics.
    """

    def __init__(
        self,
        arm_names: List[str],
        alpha: float = 0.51,
        beta: float = 1.0,
        n_min: int = 0,
        statistics_tracker: Optional[StatisticsTrackerBase] = None,
    ):
        """Initialize MDB policy.

        Args:
            arm_names: List of arm names.
            alpha: UCB exploration parameter (must be > 0.5).
            beta: Multiplier for Set F computation (must be >= 1.0).
            n_min: Grace period rounds before elimination begins.
            statistics_tracker: Optional tracker for W/N management.
        """
        super().__init__(arm_names, statistics_tracker=statistics_tracker)

        if alpha <= 0.5:
            raise ValueError("alpha must be > 0.5 to work correctly")
        if beta < 1.0:
            raise ValueError("beta must be >= 1.0")
        if n_min < 0:
            raise ValueError("n_min must be >= 0")

        self.alpha = alpha
        self.beta = beta
        self.n_min = n_min

    def _compute_ucb_matrix(self, beta_multiplier=1.0):
        log_t = np.log(max(self.t, 1))
        explored = self.N > 0
        ucb = np.ones((self.K, self.K))

        if np.any(explored):
            empirical = np.divide(
                self.W, self.N,
                out=np.zeros((self.K, self.K), dtype=float),
                where=explored
            )
            exploration = np.sqrt(
                np.divide(
                    beta_multiplier * self.alpha * log_t,
                    self.N,
                    out=np.zeros((self.K, self.K), dtype=float),
                    where=explored,
                )
            )
            ucb = np.where(explored, empirical + exploration, 1.0)

        np.fill_diagonal(ucb, np.inf)
        return ucb

    def _compute_qualifying_set_vectorized(self, beta_multiplier):
        ucb_matrix = self._compute_ucb_matrix(beta_multiplier)
        min_ucb_per_arm = np.min(ucb_matrix, axis=1)
        qualifying = np.where(min_ucb_per_arm >= 0.5)[0]
        return set(qualifying.tolist())

    def compute_set_E(self):
        return self._compute_qualifying_set_vectorized(beta_multiplier=1.0)

    def compute_set_F(self):
        if self.n_min > 0 and self.t < self.n_min:
            return set(range(self.K))
        return self._compute_qualifying_set_vectorized(beta_multiplier=self.beta)

    def select_arms(self):
        if self.n_min > 0 and self.t < self.n_min:
            return [self._idx_to_name(i) for i in range(self.K)]

        E = self.compute_set_E()
        F = self.compute_set_F()

        if len(E) == 1:
            selected_indices = E
        else:
            selected_indices = F if F else set(range(self.K))

        return [self._idx_to_name(i) for i in sorted(selected_indices)]

    def get_statistics(self):
        stats = super().get_statistics()
        E = self.compute_set_E()
        F = self.compute_set_F()
        stats.update({
            "set_E": [self._idx_to_name(i) for i in E],
            "set_F": [self._idx_to_name(i) for i in F],
            "alpha": self.alpha,
            "beta": self.beta,
            "n_min": self.n_min,
            "in_grace_period": self.n_min > 0 and self.t < self.n_min,
        })
        return stats
