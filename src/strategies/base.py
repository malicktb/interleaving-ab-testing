"""
Base class for Bandit Strategies (The Logic).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Set

import numpy as np


class BaseStrategy(ABC):
    """Base class for the Bandit Algorithm.

    The Strategy acts as the "Store Manager." Its jobs are:
    1. Keep a scoreboard of how often each model wins (W) and competes (N).
    2. Decide which models get to participate in the next round.

    Attributes:
        arm_names: List of available models (e.g., 'Linear', 'Popularity').
        K: Total number of models.
        t: Total number of rounds played so far.
        W: Win Matrix. W[i,j] is how many times Model i beat Model j.
        N: Comparison Matrix. N[i,j] is how many times i and j faced off.
    """

    def __init__(self, arm_names: List[str]):
        self.arm_names = arm_names
        self.K = len(arm_names)
        self.t = 0

        # Initialize the scoreboard (Zeros)
        self.W = np.zeros((self.K, self.K))  # Wins
        self.N = np.zeros((self.K, self.K))  # Comparisons

    def _name_to_idx(self, name: str) -> int:
        """Helper: Convert name ('Linear') to index (0)."""
        return self.arm_names.index(name)

    def _idx_to_name(self, idx: int) -> str:
        """Helper: Convert index (0) to name ('Linear')."""
        return self.arm_names[idx]

    @abstractmethod
    def select_arms(self) -> Set[str]:
        """Decide which models compete in the next round.

        Returns:
            A set of arm names (e.g., {'Linear', 'Random'}).
        """
        pass

    def update(self, winner: Optional[str], participants: Set[str]) -> None:
        """Update the scoreboard after a user interaction.

        Args:
            winner: The model that generated the clicked item (or None).
            participants: The models that were shown on the screen.
        """
        self.t += 1
        participant_list = list(participants)

        # 1. Record that these models competed against each other
        # In Multileaving, everyone on the slate fights everyone else.
        for i_idx, name_i in enumerate(participant_list):
            for name_j in participant_list[i_idx + 1:]:
                i = self._name_to_idx(name_i)
                j = self._name_to_idx(name_j)
                self.N[i, j] += 1
                self.N[j, i] += 1

        # 2. If there was a click, give the winner points
        if winner is not None:
            winner_idx = self._name_to_idx(winner)
            for name in participants:
                if name != winner:
                    loser_idx = self._name_to_idx(name)
                    # Winner beats every other model on the slate
                    self.W[winner_idx, loser_idx] += 1

    def get_win_rate(self, i: int, j: int) -> float:
        """Calculate how often Model i beats Model j (0.0 to 1.0)."""
        if self.N[i, j] == 0:
            return 0.5  # No data yet, assume they are equal
        return self.W[i, j] / self.N[i, j]

    def get_statistics(self) -> dict:
        """Return the current scoreboard."""
        return {
            "t": self.t,
            "W": self.W.tolist(),
            "N": self.N.tolist(),
        }