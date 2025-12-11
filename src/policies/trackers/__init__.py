"""Statistics trackers for pairwise comparison policies.

This module provides different strategies for tracking W/N matrices:
    - cumulative: All-time statistics (default, backward compatible)
    - discounted: Exponential decay for non-stationary arms (γ=0.995)
"""

from typing import Dict, List, Type

from .tracker_base import StatisticsTrackerBase
from .cumulative import CumulativeStatisticsTracker
from .discounted import DiscountedStatisticsTracker

# Registry of available trackers
TRACKER_REGISTRY: Dict[str, Type[StatisticsTrackerBase]] = {
    "cumulative": CumulativeStatisticsTracker,
    "discounted": DiscountedStatisticsTracker,
}


def create_statistics_tracker(
    tracker_type: str,
    arm_names: List[str],
    **kwargs
) -> StatisticsTrackerBase:
    """Factory function to create statistics tracker instances.

    Args:
        tracker_type: Name of the tracker. One of:
            - "cumulative": All-time statistics (default)
            - "discounted": Exponential decay for non-stationary arms
        arm_names: List of arm names for the tracker.
        **kwargs: Tracker-specific parameters.
            For discounted: discount_factor (float) - decay γ ∈ (0, 1), default 0.995

    Returns:
        Configured statistics tracker instance.

    Raises:
        ValueError: If tracker_type is not recognized.
    """
    if tracker_type not in TRACKER_REGISTRY:
        available = list(TRACKER_REGISTRY.keys())
        raise ValueError(
            f"Unknown tracker type: {tracker_type}. "
            f"Available: {available}"
        )

    tracker_class = TRACKER_REGISTRY[tracker_type]

    if tracker_type == "discounted":
        return tracker_class(
            arm_names=arm_names,
            discount_factor=kwargs.get("discount_factor", 0.995),
        )
    else:
        return tracker_class(arm_names=arm_names)


__all__ = [
    "StatisticsTrackerBase",
    "CumulativeStatisticsTracker",
    "DiscountedStatisticsTracker",
    "create_statistics_tracker",
    "TRACKER_REGISTRY",
]
