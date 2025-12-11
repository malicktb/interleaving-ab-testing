"""Click models for user behavior simulation.

This module provides click model implementations:
- PositionBasedModel: P(click) = P(examine|pos) * P(click|rel)
- CascadeModel: Click first relevant doc, then stop
- NoisyUserModel: Cascade with noise and false negatives
"""

from simulation.click_models.pbm import PositionBasedModel
from simulation.click_models.cascade import CascadeModel
from simulation.click_models.noisy import NoisyUserModel

__all__ = [
    "PositionBasedModel",
    "CascadeModel",
    "NoisyUserModel",
]
