"""Source module for H-MDB framework.

This module provides modeling components:
- arms: Ranking algorithms
- policies: Bandit policies
- multileaving: Slate construction and attribution
- clustering: H-MDB clustering

For simulation components, see the simulation package.
For shared base classes and utilities, see the core package.
"""

from .arms import (
    BaseArm,
    RandomArm,
    XGBoostArm,
    LinUCBArm,
    LinearTSArm,
    GroundTruthArm,
    SingleFeatureArm,
)
from .multileaving import interleave
from .policies import (
    BasePolicy,
    MDBPolicy,
    UniformPolicy,
    FixedPolicy,
    SingleArmThompsonSamplingPolicy,
)

__all__ = [
    # Arms
    "BaseArm",
    "RandomArm",
    "XGBoostArm",
    "LinUCBArm",
    "LinearTSArm",
    "GroundTruthArm",
    "SingleFeatureArm",
    # Multileaving
    "interleave",
    # Policies
    "BasePolicy",
    "MDBPolicy",
    "UniformPolicy",
    "FixedPolicy",
    "SingleArmThompsonSamplingPolicy",
]
