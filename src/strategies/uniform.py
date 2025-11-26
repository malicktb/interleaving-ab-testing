"""
Uniform baseline strategy for comparison.

This strategy always selects all arms, providing a pure exploration baseline
to compare against the UCB adaptive strategy.
"""

from .base import BaseStrategy


class UniformStrategy(BaseStrategy):
    """Always select all arms (pure exploration baseline).

    This strategy does not learn or adapt - it simply tests all arms
    in every round. Used as a baseline to demonstrate the efficiency
    gains of the UCB Multi-Dueling Bandits algorithm.
    """

    def select_arms(self):
        """Return all arm names.

        Returns:
            Set of all arm names (no filtering or learning).
        """
        return set(self.arm_names)
