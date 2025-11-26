"""
Base class for Bandit Strategies (The Logic).
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseStrategy(ABC):
    """Base class for the Bandit Algorithm.

    The Strategy acts as the "Store Manager." Its jobs are:
    1. Keep a scoreboard of how often each arm wins (W) and competes (N).
    2. Decide which arms get to participate in the next round.

    Attributes:
        arm_names: List of available arms (e.g., 'Linear', 'Popularity').
        K: Total number of arms.
        t: Total number of rounds played so far.
        W: Win Matrix. W[i,j] is how many times Arm i beat Arm j.
        N: Comparison Matrix. N[i,j] is how many times i and j faced off.
    """

    def __init__(self, arm_names):
        """Initialize the strategy.

        Args:
            arm_names: List of arm names to track.
        """
        self.arm_names = arm_names
        self.K = len(arm_names)
        self.t = 0

        # Initialize the scoreboard (Zeros)
        self.W = np.zeros((self.K, self.K))  # Wins
        self.N = np.zeros((self.K, self.K))  # Comparisons

    def _name_to_idx(self, name):
        """Helper: Convert name ('Linear') to index (0)."""
        return self.arm_names.index(name)

    def _idx_to_name(self, idx):
        """Helper: Convert index (0) to name ('Linear')."""
        return self.arm_names[idx]

    @abstractmethod
    def select_arms(self):
        """Decide which arms compete in the next round.

        Returns:
            A set of arm names (e.g., {'Linear', 'Random'}).
        """
        pass

    def update(self, winner, participants):
        """Update the scoreboard after a user interaction.

        Args:
            winner: The arm that generated the clicked item (or None).
            participants: The arms that were shown on the screen.
        """
        self.t += 1
        participant_list = list(participants)

        # 1. Record that these arms competed against each other
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
                    # Winner beats every other arm on the slate
                    self.W[winner_idx, loser_idx] += 1

    def get_win_rate(self, i, j):
        """Calculate how often Arm i beats Arm j.

        Args:
            i: Index of the first arm.
            j: Index of the second arm.

        Returns:
            Win rate from 0.0 to 1.0 (0.5 if no data).
        """
        if self.N[i, j] == 0:
            return 0.5  # No data yet, assume they are equal
        return self.W[i, j] / self.N[i, j]

    def get_statistics(self):
        """Return the current scoreboard.

        Returns:
            Dictionary with 't', 'W', and 'N' matrices.
        """
        return {
            "t": self.t,
            "W": self.W.tolist(),
            "N": self.N.tolist(),
        }