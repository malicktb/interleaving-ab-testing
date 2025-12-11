"""Base class for click simulation models.

Click models simulate user clicking behavior on ranked result slates,
providing the feedback signal for online ranker evaluation.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class ClickSimulator(ABC):
    """Abstract base class for user click simulation.

    Click simulators model how users interact with a ranked slate of documents.
    Different models capture different aspects of user behavior:
    - PositionBasedModel: Examination probability decreases with position
    - CascadeModel: User scans top-down, clicks first relevant doc
    - NoisyUserModel: Cascade with random noise and false negatives

    Subclasses must implement:
    - simulate(): Generate click positions for a slate
    """

    @abstractmethod
    def simulate(
        self,
        slate: List[int],
        relevance: np.ndarray,
        rng: np.random.Generator,
    ) -> List[int]:
        """Simulate user clicks on a slate.

        Args:
            slate: List of document indices in display order.
            relevance: Relevance grades for all documents.
            rng: Random number generator for stochastic simulation.

        Returns:
            List of clicked positions (0-indexed into slate).
        """
        pass

