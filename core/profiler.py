"""Profiler for tracking inference costs and query counts.

This module provides a thread-safe singleton Profiler for tracking:
- Total inference calls (arm.rank() invocations)
- Per-arm inference counts
- Query counts

The Profiler uses instance-level state (not class-level) to avoid
mutable default gotchas and provide clear singleton semantics.
"""

import threading
from typing import Dict, Any, Optional


class Profiler:
    """Thread-safe singleton profiler for inference cost tracking.

    Usage:
        profiler = Profiler()  # Returns singleton instance
        profiler.increment_inference(1, arm_name="xgboost")
        profiler.increment_query()
        print(profiler.get_statistics())
        profiler.reset()  # Clear all counters
    """

    _instance: Optional['Profiler'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'Profiler':
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    # Initialize instance state here (not in __init__)
                    # to ensure it only happens once
                    instance._total_inferences = 0
                    instance._total_queries = 0
                    instance._arm_inference_counts: Dict[str, int] = {}
                    cls._instance = instance
        return cls._instance

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        n_arms = len(self._arm_inference_counts)
        return (
            f"Profiler(queries={self._total_queries}, "
            f"inferences={self._total_inferences}, "
            f"arms_tracked={n_arms})"
        )

    def increment_inference(self, count: int = 1, arm_name: Optional[str] = None) -> None:
        """Increment inference counter.

        Args:
            count: Number of inferences to add (default 1).
            arm_name: Optional arm name for per-arm tracking.
        """
        with self._lock:
            self._total_inferences += count
            if arm_name:
                self._arm_inference_counts[arm_name] = (
                    self._arm_inference_counts.get(arm_name, 0) + count
                )

    def increment_query(self) -> None:
        """Increment query counter by 1.

        Call this once per query/step processed, regardless of
        how many arms were evaluated.
        """
        with self._lock:
            self._total_queries += 1

    def get_total_cost(self) -> int:
        """Get total inference count.

        Returns:
            Total number of inference calls.
        """
        return self._total_inferences

    def get_avg_arms_per_query(self) -> float:
        """Get average number of arms evaluated per query.

        Returns:
            Average arms per query, or 0.0 if no queries.
        """
        if self._total_queries == 0:
            return 0.0
        return self._total_inferences / self._total_queries

    def get_arm_costs(self) -> Dict[str, int]:
        """Get copy of per-arm inference counts.

        Returns:
            Dict of arm_name -> inference count.
        """
        with self._lock:
            return self._arm_inference_counts.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive profiler statistics.

        Returns:
            Dict with total_queries, total_inferences, avg_arms_per_query,
            and per-arm costs.
        """
        return {
            "total_queries": self._total_queries,
            "total_inferences": self._total_inferences,
            "avg_arms_per_query": self.get_avg_arms_per_query(),
            "arm_costs": self.get_arm_costs(),
        }

    def reset(self) -> None:
        """Reset all counters to initial state.

        Creates fresh containers for all mutable state. Safe to call
        between experiments or episodes.
        """
        with self._lock:
            self._total_inferences = 0
            self._total_queries = 0
            # Create new dict to avoid any lingering references
            self._arm_inference_counts = {}
