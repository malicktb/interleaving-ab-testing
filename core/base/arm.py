"""Base class for ranking arms.

Arms are ranking algorithms that produce document orderings for queries.
They can be static (trained once) or learning (update from feedback).
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseArm(ABC):
    """Abstract base class for ranking arms.

    All ranking algorithms (XGBoost, LinUCB, etc.) inherit from this class.
    Provides common interface for training, ranking, and online updates.
    """

    def __init__(self, name: str):
        """Initialize arm with a name.

        Args:
            name: Unique identifier for this arm.
        """
        self.name = name
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Whether the arm has been trained."""
        return self._is_trained

    @abstractmethod
    def train(self, train_records: List) -> None:
        """Train the arm on historical data.

        Args:
            train_records: List of QueryRecord for training.
        """
        pass

    @abstractmethod
    def rank(self, record) -> np.ndarray:
        """Produce a ranking for a query.

        Args:
            record: QueryRecord with features and metadata.

        Returns:
            Array of document indices in ranked order.
        """
        pass

    def update(self, features: np.ndarray, reward: float) -> None:
        """Update arm with online feedback (optional).

        Only learning arms (LinUCB, EpsilonGreedy, etc.) implement this.
        Static arms can leave this as a no-op.

        Args:
            features: Feature vector for the clicked item.
            reward: Click/no-click signal (1.0 or 0.0).
        """
        pass
