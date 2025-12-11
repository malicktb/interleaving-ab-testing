"""Score pre-computation cache for scalable simulation.

Pre-computes and caches arm scores for all static rankers to avoid
expensive rank() calls during simulation. Learning arms are
computed live.

Memory estimate for K=100, Q=6415, D_avg=25:
  100 * 6415 * 25 * 4 bytes = ~64MB (very manageable)
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from core.types import QueryRecord


class ScoreCache:
    """Pre-computation cache for arm scores.

    Stores pre-computed rankings for static arms to enable O(1) lookup
    during simulation instead of O(K) arm.rank() calls per query.

    Structure:
        scores[arm_name][query_idx] = ranking (array of doc indices)

    Learning arms return None and must be computed live.
    """

    def __init__(self):
        """Initialize empty cache."""
        self._cache: Dict[str, Dict[int, np.ndarray]] = {}
        self._query_ids: List[str] = []
        self._query_id_to_idx: Dict[str, int] = {}
        self._cached_arms: List[str] = []
        self._is_built = False

    @property
    def is_built(self) -> bool:
        """Whether cache has been populated."""
        return self._is_built

    @property
    def cached_arm_names(self) -> List[str]:
        """List of arm names in cache."""
        return self._cached_arms.copy()

    @property
    def n_queries(self) -> int:
        """Number of queries in cache."""
        return len(self._query_ids)

    def build(
        self,
        arms: Dict[str, Any],
        records: List[QueryRecord],
        exclude_arms: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Pre-compute scores for all arms on all queries.

        Args:
            arms: Dict of arm_name -> trained arm instance.
            records: List of QueryRecord to cache.
            exclude_arms: Arm names to skip (learning arms).
            verbose: Print progress.

        Returns:
            Dict with cache statistics.
        """
        exclude_arms = exclude_arms or []

        # Store query mapping
        self._query_ids = [r.query_id for r in records]
        self._query_id_to_idx = {qid: idx for idx, qid in enumerate(self._query_ids)}

        # Filter arms to cache (exclude learning arms)
        arms_to_cache = {
            name: arm for name, arm in arms.items()
            if name not in exclude_arms
        }

        if verbose:
            print(f"[Cache] Building score cache for {len(arms_to_cache)} arms, {len(records)} queries...")
            if exclude_arms:
                print(f"[Cache] Excluded (learning arms): {exclude_arms}")

        self._cache = {}
        self._cached_arms = []

        for arm_name, arm in arms_to_cache.items():
            if verbose:
                print(f"[Cache]   Pre-computing {arm_name}...")

            self._cache[arm_name] = {}

            for idx, record in enumerate(records):
                ranking = arm.rank(record)
                # Store as int16 to save memory (max 32767 docs per query)
                self._cache[arm_name][idx] = ranking.astype(np.int16)

            self._cached_arms.append(arm_name)

        self._is_built = True

        # Compute memory usage
        total_bytes = 0
        for arm_name in self._cache:
            for idx in self._cache[arm_name]:
                total_bytes += self._cache[arm_name][idx].nbytes

        stats = {
            "n_arms": len(self._cached_arms),
            "n_queries": len(records),
            "excluded_arms": exclude_arms,
            "memory_mb": total_bytes / (1024 * 1024),
        }

        if verbose:
            print(f"[Cache] Built: {stats['n_arms']} arms, {stats['n_queries']} queries, {stats['memory_mb']:.1f} MB")

        return stats

    def get_ranking(
        self,
        arm_name: str,
        query_id: str,
    ) -> Optional[np.ndarray]:
        """Get cached ranking for an arm on a query.

        Args:
            arm_name: Name of the arm.
            query_id: Query ID.

        Returns:
            Cached ranking array, or None if not in cache.
        """
        if arm_name not in self._cache:
            return None

        query_idx = self._query_id_to_idx.get(query_id)
        if query_idx is None:
            return None

        ranking = self._cache[arm_name].get(query_idx)
        if ranking is not None:
            return ranking.astype(np.int64)  # Return as int64 for compatibility
        return None

    def get_ranking_by_idx(
        self,
        arm_name: str,
        query_idx: int,
    ) -> Optional[np.ndarray]:
        """Get cached ranking by query index (faster than by ID).

        Args:
            arm_name: Name of the arm.
            query_idx: Query index in cache.

        Returns:
            Cached ranking array, or None if not in cache.
        """
        if arm_name not in self._cache:
            return None

        ranking = self._cache[arm_name].get(query_idx)
        if ranking is not None:
            return ranking.astype(np.int64)
        return None

    def save(self, path: str) -> None:
        """Save cache to disk.

        Args:
            path: Path to save file (.pkl).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "cache": self._cache,
            "query_ids": self._query_ids,
            "cached_arms": self._cached_arms,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[Cache] Saved to {path}")

    def load(self, path: str) -> bool:
        """Load cache from disk.

        Args:
            path: Path to saved file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not os.path.exists(path):
            print(f"[Cache] File not found: {path}")
            return False

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self._cache = data["cache"]
            self._query_ids = data["query_ids"]
            self._query_id_to_idx = {qid: idx for idx, qid in enumerate(self._query_ids)}
            self._cached_arms = data["cached_arms"]
            self._is_built = True

            print(f"[Cache] Loaded: {len(self._cached_arms)} arms, {len(self._query_ids)} queries")
            return True

        except Exception as e:
            print(f"[Cache] Load failed: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache info.
        """
        if not self._is_built:
            return {"is_built": False}

        total_bytes = sum(
            self._cache[arm][idx].nbytes
            for arm in self._cache
            for idx in self._cache[arm]
        )

        return {
            "is_built": True,
            "n_arms": len(self._cached_arms),
            "n_queries": len(self._query_ids),
            "cached_arms": self._cached_arms,
            "memory_mb": total_bytes / (1024 * 1024),
        }


def precompute_scores(
    arms: Dict[str, Any],
    data_loader: Any,
    exclude_arms: Optional[List[str]] = None,
    cache_path: Optional[str] = None,
    max_queries: Optional[int] = None,
) -> Tuple[ScoreCache, List[QueryRecord]]:
    """Convenience function to build score cache.

    Args:
        arms: Dict of arm_name -> trained arm.
        data_loader: DataLoader with iter_test_records().
        exclude_arms: Arms to exclude from caching.
        cache_path: Optional path to save cache.
        max_queries: Optional limit on queries to cache.

    Returns:
        Tuple of (ScoreCache, list of test records).
    """
    # Load all test records
    records = list(data_loader.iter_test_records())
    if max_queries:
        records = records[:max_queries]

    # Build cache
    cache = ScoreCache()
    cache.build(
        arms=arms,
        records=records,
        exclude_arms=exclude_arms,
    )

    # Optionally save
    if cache_path:
        cache.save(cache_path)

    return cache, records
