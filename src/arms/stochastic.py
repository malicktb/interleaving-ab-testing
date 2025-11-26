"""
Random ranking policy (arm).
"""

import numpy as np

from .base import BaseArm


class StochasticArm(BaseArm):
    """Arm that randomly shuffles items.

    This acts as a "Control Group" or baseline. Since it just guesses randomly,
    a good Bandit Strategy should quickly learn to ignore this arm in favor
    of the smarter ones.

    Attributes:
        rng: Random number generator.
    """

    def __init__(self, name="stochastic", random_state=42):
        """Initialize the stochastic arm.

        Args:
            name: Unique identifier for this arm.
            random_state: Seed for the random number generator.
        """
        super().__init__(name)
        self.rng = np.random.default_rng(random_state)

    def train(self, train_data=None):
        """Do nothing.
        
        Random guessing does not require learning from historical data.
        """
        self._is_trained = True

    def rank(self, record):
        """Return a random permutation of item indices.

        Args:
            record: Dictionary containing item data including 'labels'.

        Returns:
            Array of randomly shuffled item indices.
        """
        num_items = len(record["labels"])
        return self.rng.permutation(num_items)
