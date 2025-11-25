"""
Random ranking policy (arm).
"""

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseArm


class StochasticArm(BaseArm):
    """Arm that randomly shuffles items.

    This acts as a "Control Group" or baseline. Since it just guesses randomly,
    a good Bandit Strategy should quickly learn to ignore this arm in favor
    of the smarter ones.

    Attributes:
        rng: Random number generator.
    """

    def __init__(self, name: str = "stochastic", random_state: Optional[int] = 42):
        super().__init__(name)
        self.rng = np.random.default_rng(random_state)

    def train(self, train_data: pd.DataFrame = None) -> None:
        """Do nothing.
        
        Random guessing does not require learning from historical data.
        """
        self._is_trained = True

    def rank(self, record: dict) -> np.ndarray:
        """Return a random list of item indices (0 to 29)."""
        return self.rng.permutation(30)
