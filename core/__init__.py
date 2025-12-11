"""Core module: Shared base classes, types, and utilities.

This module provides foundational components used across the codebase:
- Base classes (BaseArm, BasePolicy, etc.)
- Shared types (QueryRecord, AttributionResult, ClusteringResult)
- Core metrics (DCG, NDCG, RegretTelemetry)
- Profiler (inference cost tracking)

Example usage:
    from core.base import BaseArm, BasePolicy
    from core.types import QueryRecord, AttributionResult
    from core.metrics import compute_ndcg, RegretTelemetry
    from core import Profiler
"""

# Base classes
from core.base import (
    BaseArm,
    BasePolicy,
    StatisticsTrackerBase,
    BaseAttributionStrategy,
    ClickSimulator,
)

# Types
from core.types import (
    QueryRecord,
    AttributionResult,
    ClusteringResult,
)

# Metrics
from core.metrics import (
    compute_dcg,
    compute_ndcg,
    RegretTelemetry,
)

# Profiler
from core.profiler import Profiler

__all__ = [
    # Base classes
    "BaseArm",
    "BasePolicy",
    "StatisticsTrackerBase",
    "BaseAttributionStrategy",
    "ClickSimulator",
    # Types
    "QueryRecord",
    "AttributionResult",
    "ClusteringResult",
    # Metrics
    "compute_dcg",
    "compute_ndcg",
    "RegretTelemetry",
    # Profiler
    "Profiler",
]
