"""Multileaving strategies aligned with published algorithms."""

from typing import Callable, Dict, List, Tuple
import numpy as np

from .team_draft import interleave as team_draft_interleave


def create_multileaver(
    scheme: str = "team_draft",
) -> Callable:
    """Factory for multileaving strategy callables.

    Args:
        scheme: Multileaving scheme to use. Currently only "team_draft" is supported.

    Returns:
        Callable that performs multileaving.

    Raises:
        ValueError: If scheme is not recognized.
    """
    if scheme == "team_draft":
        return team_draft_interleave
    raise ValueError(f"Unknown multileaving scheme: {scheme}")
