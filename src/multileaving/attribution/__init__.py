"""Attribution strategies for credit assignment in multileaving.

This module provides different strategies for transforming slate-level user
clicks into pairwise arm outcomes that bandit policies can consume.

Available strategies:
    - team_draft_legacy: First click determines sole winner (default, backward compatible)
"""

from typing import Dict, Type

from .attribution_base import AttributionResult, BaseAttributionStrategy
from .team_draft import TeamDraftAttribution

# Registry of available attribution strategies
ATTRIBUTION_REGISTRY: Dict[str, Type[BaseAttributionStrategy]] = {
    "team_draft_legacy": TeamDraftAttribution,
}


def create_attribution_strategy(
    strategy_type: str = "team_draft_legacy",
    **kwargs
) -> BaseAttributionStrategy:
    """Factory function to create attribution strategy instances.

    Args:
        strategy_type: Name of the strategy. Currently only:
            - "team_draft_legacy": First-click-wins (default)
        **kwargs: Strategy-specific parameters (unused for team_draft_legacy).

    Returns:
        Configured attribution strategy instance.

    Raises:
        ValueError: If strategy_type is not recognized.
    """
    if strategy_type not in ATTRIBUTION_REGISTRY:
        available = list(ATTRIBUTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown attribution strategy: {strategy_type}. "
            f"Available: {available}"
        )

    strategy_class = ATTRIBUTION_REGISTRY[strategy_type]
    return strategy_class()


__all__ = [
    # Base classes
    "AttributionResult",
    "BaseAttributionStrategy",
    # Implementations
    "TeamDraftAttribution",
    # Factory and registry
    "create_attribution_strategy",
    "ATTRIBUTION_REGISTRY",
]
