"""Core simulation engine for H-MDB experiments.

The Simulator orchestrates:
- Phase 1: Offline arm training
- Phase 2: Online simulation with policy selection, multileaving,
  click simulation, attribution, and policy updates

Supports optional ScoreCache for O(1) ranking lookup (K=100 scalability).
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List

from core.base import BaseAttributionStrategy
from core import Profiler
from core.metrics import RegretTelemetry
from core.types import QueryRecord

from analysis.tracking import (
    SuccessRateTracker,
    CostReductionCalculator,
)
from simulation.data.cache import ScoreCache

# Multileaving imports
from src.multileaving import interleave, TeamDraftAttribution


class Simulator:
    """Core simulation engine for H-MDB experiments.

    Orchestrates the full experiment pipeline:
    1. Train arms on historical data (Phase 1)
    2. Run online simulation (Phase 2)
    """

    def __init__(
        self,
        arms: Dict[str, Any],
        policy: Any,
        data_loader: Any,
        click_model: Any,
        ground_truth: Any = None,
        slate_size: int = 10,
        random_seed: int = 42,
        attribution_strategy: Optional[BaseAttributionStrategy] = None,
        multileaver=None,
        hierarchical: bool = False,
        score_cache: Optional[ScoreCache] = None,
    ):
        """Initialize simulator.

        Args:
            arms: Dict of arm_name -> arm instance.
            policy: Policy for arm selection.
            data_loader: DataLoader for train/test data.
            click_model: Click model for user simulation.
            ground_truth: Optional oracle arm for regret computation.
            slate_size: Documents per result page.
            random_seed: Random seed for reproducibility.
            attribution_strategy: Strategy for credit assignment.
            multileaver: Multileaving function.
            hierarchical: Whether using hierarchical evaluation.
            score_cache: Optional ScoreCache for O(1) ranking lookup.
        """
        self.arms = arms
        self.policy = policy
        self.data_loader = data_loader
        self.click_model = click_model
        self.ground_truth = ground_truth
        self.slate_size = slate_size
        self.rng = np.random.default_rng(random_seed)

        # Use provided attribution strategy or default to Team Draft
        self.attribution_strategy = attribution_strategy or TeamDraftAttribution()
        self.multileaver = multileaver or interleave

        self.history: Optional[RegretTelemetry] = None
        self._iteration = 0

        self.profiler = Profiler()

        # Phase 7+8 metrics for hierarchical evaluation
        self.hierarchical = hierarchical
        self.success_tracker: Optional[SuccessRateTracker] = None
        self.cost_calculator: Optional[CostReductionCalculator] = None

        # Score cache for K=100 scalability
        self.score_cache = score_cache
        self._cache_hits = 0
        self._cache_misses = 0

        # Fix 1: Track logical evaluations (total arms evaluated per query)
        # This counts ALL arm evaluations, regardless of cache status
        self._logical_evaluations = 0

    def reset(self) -> Dict[str, Any]:
        """Reset simulator state.

        Returns:
            Dict with initial iteration count.
        """
        self.history = RegretTelemetry(arm_names=list(self.arms.keys()))
        self._iteration = 0
        self.profiler.reset()

        # Initialize Phase 7+8 trackers
        self.success_tracker = SuccessRateTracker(threshold_pct=0.01)
        self.cost_calculator = None

        # Reset cache statistics
        self._cache_hits = 0
        self._cache_misses = 0

        # Reset logical evaluations counter
        self._logical_evaluations = 0

        return {"iteration": 0}

    def initialize_cost_calculator(
        self,
        total_arms: int,
        level1_arms: int,
        level2_arms: int,
    ) -> None:
        """Initialize the cost reduction calculator with arm counts.

        Should be called after hierarchical policy is set up.

        Args:
            total_arms: Total number of arms.
            level1_arms: Arms in Level 1.
            level2_arms: Arms in Level 2.
        """
        self.cost_calculator = CostReductionCalculator(
            total_arms=total_arms,
            level1_arms=level1_arms,
            level2_arms=level2_arms,
        )

    def train_arms(self, max_train_records: Optional[int] = None) -> None:
        """Train all arms on historical data (Phase 1).

        Args:
            max_train_records: Optional limit on training records.
        """
        print("Starting training phase...")
        start_time = time.time()

        if hasattr(self.data_loader, 'sample_train_records'):
            train_records = self.data_loader.sample_train_records(max_train_records)
        else:
            train_records = list(self.data_loader.iter_train_records())
            if max_train_records:
                train_records = train_records[:max_train_records]

        print(f"Collected {len(train_records)} training records")

        for name, arm in self.arms.items():
            print(f"Training {name}...")
            arm.train(train_records)

        if self.ground_truth and not self.ground_truth.is_trained:
            self.ground_truth.train(train_records)

        print(f"Training complete in {time.time() - start_time:.1f}s")

    def step(self, record: QueryRecord, query_idx: Optional[int] = None) -> Dict[str, Any]:
        """Run one simulation step.

        Args:
            record: Query record for this step.
            query_idx: Optional query index for cache lookup (faster than by ID).

        Returns:
            Dict with clicks, winner, slate, ndcg, and iteration.
        """
        if self.history is None:
            self.reset()

        self.profiler.increment_query()

        active_arm_names = self.policy.select_arms()

        # Build rankings dict with hybrid cache/live execution
        rankings = {}
        for name in active_arm_names:
            cached_ranking = None

            # Try cache lookup if available
            if self.score_cache is not None:
                if query_idx is not None:
                    cached_ranking = self.score_cache.get_ranking_by_idx(name, query_idx)
                else:
                    cached_ranking = self.score_cache.get_ranking(name, record.query_id)

            if cached_ranking is not None:
                # Cache hit - use pre-computed ranking
                rankings[name] = cached_ranking
                self._cache_hits += 1
            else:
                # Cache miss - compute live
                rankings[name] = self.arms[name].rank(record)
                self._cache_misses += 1
                self.profiler.increment_inference(1, arm_name=name)

            # Fix 1: Always count logical evaluation regardless of cache status
            # This measures the "algorithm cost" - how many arms the policy looked at
            self._logical_evaluations += 1

        slate, attribution = self.multileaver(
            rankings=rankings,
            active_arms=active_arm_names,
            slate_size=min(self.slate_size, record.num_items),
            rng=self.rng,
        )

        if not slate:
            self._iteration += 1
            return {"clicks": [], "winner": None, "slate": []}

        slate_relevance = record.relevance[slate]

        clicks = self.click_model.simulate(
            slate=list(range(len(slate))),
            relevance=slate_relevance,
            rng=self.rng,
        )

        # Use attribution strategy to compute pairwise outcomes
        attribution_result = self.attribution_strategy.compute_attribution(
            clicks=clicks,
            slate=slate,
            attribution_map=attribution,
            rankings=rankings,
            participants=active_arm_names,
        )

        winner_name = attribution_result.winner

        # Update policy using pairwise outcomes
        self.policy.update(
            winner_name,
            active_arm_names,
            pairwise_outcomes=attribution_result.pairwise_outcomes,
        )

        if clicks:
            first_click_idx = clicks[0]
            clicked_doc_idx = slate[first_click_idx]
            clicked_features = record.features[clicked_doc_idx]

            # Update all arms with click feedback
            for arm in self.arms.values():
                arm.update(clicked_features, reward=1.0)

        # Use None as sentinel for "no ground truth" vs 0.0 meaning "zero quality"
        slate_ndcg = None

        self.history.record_iteration(active_arm_names, winner_name)

        # Track level transitions for hierarchical policy
        if self.hierarchical and hasattr(self.policy, 'current_level'):
            current_level = self.policy.current_level
            if self.cost_calculator:
                self.cost_calculator.record_round(current_level, len(active_arm_names))

        if self.ground_truth:
            optimal_ranking = self.ground_truth.rank(record)
            optimal_ndcg = self.ground_truth.get_ndcg(
                record, optimal_ranking, k=self.slate_size
            )
            slate_ndcg = self.ground_truth.get_ndcg(
                record, np.array(slate), k=len(slate)
            )

            self.history.record_ndcg_regret(optimal_ndcg, slate_ndcg)

        self._iteration += 1

        return {
            "clicks": clicks,
            "winner": winner_name,
            "slate": slate,
            "ndcg": slate_ndcg,
            "iteration": self._iteration,
        }

    def run_episode(
        self,
        max_iterations: int = None,
        log_interval: int = 1000,
        test_records: Optional[List[QueryRecord]] = None,
    ) -> RegretTelemetry:
        """Run full simulation episode.

        Args:
            max_iterations: Optional iteration limit.
            log_interval: Iterations between progress logs.
            test_records: Optional pre-loaded test records (enables cache by index).

        Returns:
            RegretTelemetry with results.
        """
        print("Starting simulation loop...")
        self.reset()

        # Use provided records or iterate from data_loader
        if test_records is not None:
            # Fix 2: Cycle through pre-loaded records if max_iterations exceeds list length
            # This allows runs to exceed the dataset size while maintaining cache-by-index
            def cycling_records_iterator(records, max_iters):
                """Cycle through records indefinitely, yielding (index % len, record)."""
                n_records = len(records)
                for i in range(max_iters if max_iters else float('inf')):
                    idx = i % n_records
                    yield idx, records[idx]

            if max_iterations and max_iterations > len(test_records):
                print(f"  [Looping] {max_iterations} rounds > {len(test_records)} records, cycling dataset")
                records_iter = cycling_records_iterator(test_records, max_iterations)
            else:
                # Original behavior - direct enumeration
                records_iter = enumerate(test_records)
        else:
            # Fix 2: Use infinite iterator to allow runs beyond dataset size
            # This loops through the dataset repeatedly with shuffling
            if hasattr(self.data_loader, 'iter_test_records_infinite'):
                records_iter = ((None, r) for r in self.data_loader.iter_test_records_infinite())
            else:
                # Fall back to original iterator (will stop at dataset end)
                records_iter = ((None, r) for r in self.data_loader.iter_test_records())

        for query_idx, record in records_iter:
            if max_iterations and self._iteration >= max_iterations:
                break

            self.step(record, query_idx=query_idx)

            if self._iteration % log_interval == 0:
                self._log_progress()

        print(f"Simulation complete: {self._iteration} iterations.")
        if self.score_cache is not None:
            total = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total * 100 if total > 0 else 0
            print(f"  Cache: {self._cache_hits}/{total} hits ({hit_rate:.1f}%)")
        return self.history

    def _log_progress(self) -> None:
        """Log current progress."""
        stats = self.policy.get_statistics()
        set_e = stats.get("set_E", [])
        in_grace = stats.get("in_grace_period", False)

        regret = self.history.total_ndcg_regret

        grace_str = " [GRACE]" if in_grace else ""

        print(
            f"  Iter {self._iteration}: "
            f"CumRegret={regret:.1f}, "
            f"Cost={self.profiler.get_total_cost()}, "
            f"Winners={set_e}{grace_str}"
        )

    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive results.

        Returns:
            Dict with metrics, rates, stats, and history.
        """
        if self.history is None:
            return {}

        summary = self.history.get_summary()

        results = {
            "metrics": summary,
            "selection_rates": self.history.get_arm_selection_rates(),
            "win_rates": self.history.get_arm_win_rates(),
            "policy_stats": self.policy.get_statistics(),
            "inference_stats": self.profiler.get_statistics(),
        }

        results["history"] = {
            "cumulative_ndcg_regret": self.history.cumulative_ndcg_regret_history
        }

        # Include Phase 7+8 metrics
        if self.success_tracker:
            results["success_metrics"] = self.success_tracker.get_summary()

        if self.cost_calculator:
            results["cost_metrics"] = self.cost_calculator.get_summary()

        # Include hierarchical info
        if self.hierarchical:
            results["hierarchical"] = {
                "enabled": True,
            }

        # Include cache statistics
        if self.score_cache is not None:
            total_lookups = self._cache_hits + self._cache_misses
            results["cache_stats"] = {
                "enabled": True,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "total_lookups": total_lookups,
                "hit_rate": self._cache_hits / total_lookups if total_lookups > 0 else 0.0,
                "cached_arms": self.score_cache.cached_arm_names,
                "n_cached_queries": self.score_cache.n_queries,
            }

        # Fix 1: Add logical evaluation cost metric
        # This is the total number of arm rankings the algorithm "looked at"
        n_queries = self.history.num_iterations if self.history else 0
        results["logical_cost"] = {
            "total_evaluations": self._logical_evaluations,
            "avg_arms_per_query": self._logical_evaluations / n_queries if n_queries > 0 else 0.0,
            "total_queries": n_queries,
        }

        return results
