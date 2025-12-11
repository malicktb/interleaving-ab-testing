"""Baseline policies for comparison against MDB."""

from typing import List, Optional, Dict
import numpy as np

from .base import BasePolicy
from .trackers import StatisticsTrackerBase


class UniformPolicy(BasePolicy):
    """Always select all arms (no elimination)."""

    def select_arms(self) -> List[str]:
        return list(self.arm_names)


class FixedPolicy(BasePolicy):
    """Always select a single fixed arm."""

    def __init__(
        self,
        arm_names: List[str],
        fixed_arm_name: str,
        statistics_tracker: Optional[StatisticsTrackerBase] = None,
    ):
        super().__init__(arm_names, statistics_tracker=statistics_tracker)
        if fixed_arm_name not in arm_names:
            raise ValueError(f"Arm '{fixed_arm_name}' not in {arm_names}")
        self.fixed_arm = fixed_arm_name

    def select_arms(self) -> List[str]:
        return [self.fixed_arm]


class SingleArmThompsonSamplingPolicy(BasePolicy):
    """Traditional A/B testing simulation with Thompson Sampling.

    Selects a single arm per round using Thompson Sampling (Beta posteriors).
    Updates only the played arm's posterior based on win/loss.
    """

    def __init__(
        self,
        arm_names: List[str],
        seed: int = 42,
        statistics_tracker: Optional[StatisticsTrackerBase] = None,
    ):
        super().__init__(arm_names, statistics_tracker=statistics_tracker)
        self.rng = np.random.default_rng(seed)
        self.alphas = np.ones(self.K)
        self.betas = np.ones(self.K)

    def select_arms(self) -> List[str]:
        samples = self.rng.beta(self.alphas, self.betas)
        best_idx = np.argmax(samples)
        return [self.arm_names[best_idx]]

    def update(
        self,
        winner: Optional[str],
        participants: List[str],
        pairwise_outcomes: Optional[Dict[tuple, float]] = None,
    ) -> None:
        # Call parent to update tracker (increments t, updates W/N)
        super().update(winner, participants, pairwise_outcomes=pairwise_outcomes)

        # Update Thompson Sampling posteriors
        played_arm = participants[0]
        played_idx = self.arm_names.index(played_arm)

        if winner == played_arm:
            self.alphas[played_idx] += 1
        else:
            self.betas[played_idx] += 1

    def get_statistics(self) -> dict:
        stats = super().get_statistics()
        stats["alphas"] = self.alphas.tolist()
        stats["betas"] = self.betas.tolist()
        estimates = self.alphas / (self.alphas + self.betas)
        stats["estimated_rates"] = {
            name: float(estimates[i])
            for i, name in enumerate(self.arm_names)
        }
        return stats
