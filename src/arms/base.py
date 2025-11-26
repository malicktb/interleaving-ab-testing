"""
Base class for ranking models.
"""

from abc import ABC, abstractmethod


class BaseArm(ABC):
    """Base class for a ranking model (an 'Arm').

    Each Arm is a specific strategy for sorting items. The Bandit Strategy
    decides which Arm to use for a given user.

    Attributes:
        name: Unique name for this model.
        is_trained: True if the model is ready to use.
    """

    def __init__(self, name):
        self.name = name
        self._is_trained = False

    @property
    def is_trained(self):
        """Check if model is trained."""
        return self._is_trained

    @abstractmethod
    def train(self, train_data):
        """Train the model using historical data.

        Args:
            train_data: DataFrame containing item features and user labels.
        """
        pass

    @abstractmethod
    def rank(self, record):
        """Rank the items for a single user request.

        Args:
            record: Dictionary containing item features and metadata.

        Returns:
            Array of item indices, sorted from best to worst.
        """
        pass