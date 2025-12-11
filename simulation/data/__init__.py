"""Data loading and preprocessing utilities.

This module provides:
- DataLoader: Load Yahoo LTR dataset from Parquet
- ScoreCache: Pre-computation cache for arm scores (scalability)
- Format converters: convert_letor_to_parquet, convert_file
"""

from simulation.data.loader import DataLoader, REQUIRED_COLUMNS
from simulation.data.cache import ScoreCache, precompute_scores
from simulation.data.converter import (
    convert_letor_to_parquet,
    convert_file,
    parse_letor_line,
    densify_features,
    group_by_query,
)

__all__ = [
    "DataLoader",
    "REQUIRED_COLUMNS",
    "ScoreCache",
    "precompute_scores",
    "convert_letor_to_parquet",
    "convert_file",
    "parse_letor_line",
    "densify_features",
    "group_by_query",
]
