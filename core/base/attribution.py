"""Base classes for attribution strategies.

Attribution strategies transform slate-level user clicks into pairwise
arm outcomes that a dueling bandit policy can consume.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from core.types import AttributionResult


class BaseAttributionStrategy(ABC):
    """Abstract base class for attribution strategies.

    Attribution strategies transform slate-level user clicks into pairwise
    arm outcomes that a dueling bandit policy can consume.

    Implementations include:
    - TeamDraftAttribution: First-click-wins
    - PPMAttribution: Probabilistic pairwise preference
    - SOSMAttribution: DCG-based scoring
    """

    @abstractmethod
    def compute_attribution(
        self,
        clicks: List[int],
        slate: List[int],
        attribution_map: Dict[int, str],
        rankings: Dict[str, List[int]],
        participants: List[str],
    ) -> AttributionResult:
        """Compute pairwise outcomes from clicks.

        Args:
            clicks: List of clicked positions in slate (0-indexed).
            slate: List of doc indices in display order.
            attribution_map: Maps doc_idx to the arm that contributed it to slate.
            rankings: Original rankings from each arm (arm_name -> list of doc indices).
            participants: Arms that participated in this round.

        Returns:
            AttributionResult with pairwise outcomes and optional single winner.
        """
        pass

