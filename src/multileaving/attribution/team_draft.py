"""Team Draft attribution - first click determines sole winner."""

from typing import Dict, List, Optional, Tuple

from .attribution_base import AttributionResult, BaseAttributionStrategy


class TeamDraftAttribution(BaseAttributionStrategy):
    """First-click-wins attribution for Team Draft slates (baseline).

    For each clicked document, credit is assigned to the arm that contributed
    that document to the slate. The bandit policy treats the arm with the
    first credited click as the "winner" on that round.

    This method is simple but known to suffer from similarity bias: arms that
    often agree in their top results tend to help each other, leading to
    distorted win statistics.

    This is the default attribution scheme for backward compatibility.
    """

    def compute_attribution(
        self,
        clicks: List[int],
        slate: List[int],
        attribution_map: Dict[int, str],
        rankings: Dict[str, List[int]],
        participants: List[str],
    ) -> AttributionResult:
        """Compute attribution based on first click.

        Args:
            clicks: List of clicked positions in slate.
            slate: List of doc indices in display order.
            attribution_map: Maps doc_idx to the arm that contributed it.
            rankings: Original rankings (unused in this strategy).
            participants: Arms that participated in this round.

        Returns:
            AttributionResult with single winner beating all other participants.
        """
        if not clicks:
            return AttributionResult(
                pairwise_outcomes={},
                winner=None,
                credits={},
            )

        # First click determines winner
        first_click_pos = clicks[0]
        clicked_item = slate[first_click_pos]
        winner = attribution_map.get(clicked_item)

        if winner is None:
            return AttributionResult(
                pairwise_outcomes={},
                winner=None,
                credits={},
            )

        # Generate pairwise outcomes: winner beats all other participants
        pairwise_outcomes: Dict[Tuple[str, str], float] = {}
        for loser in participants:
            if loser != winner:
                pairwise_outcomes[(winner, loser)] = 1.0

        credits = {winner: 1.0}

        return AttributionResult(
            pairwise_outcomes=pairwise_outcomes,
            winner=winner,
            credits=credits,
        )
