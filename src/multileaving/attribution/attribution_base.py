"""Base classes for attribution strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AttributionResult:
    """Structured result from attribution computation.

    Attributes:
        pairwise_outcomes: Dict mapping (winner_arm, loser_arm) -> credit weight.
            Used by policies to update W/N matrices.
        winner: Single winner arm name for backward compatibility with
            legacy code that expects a single winner per round.
        credits: Per-arm credit scores (useful for debugging/analysis).
    """
    pairwise_outcomes: Dict[Tuple[str, str], float] = field(default_factory=dict)
    winner: Optional[str] = None
    credits: Dict[str, float] = field(default_factory=dict)


class BaseAttributionStrategy(ABC):
    """Abstract base class for attribution strategies.

    Attribution strategies transform slate-level user clicks into pairwise
    arm outcomes that a dueling bandit policy can consume.
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
