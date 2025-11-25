"""Simulation loop for Multi-Armed Bandit experiments."""

import time
from collections import defaultdict
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from ..arms.base import BaseArm
from ..arms.ranking_policies import PopularityArm
from ..config import ExperimentConfig
from ..data.loader import DataLoader
from .rewards import compute_reward
from ..interleaving.sampling import sample_slate
from ..strategies.base import BaseStrategy
from .telemetry import RegretTelemetry


class Simulation:
    """Orchestrates the Monte Carlo MAB experiment.

    Acts as the 'Game Engine' that connects:
    1. The Environment (Data + Rewards)
    2. The Controller (Bandit Strategy)
    3. The Arms (Ranking Policies)
    4. The Telemetry (History/Logs)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        arms: Dict[str, BaseArm],
        strategy: BaseStrategy,
        data_loader: DataLoader,
    ):
        self.config = config
        self.arms = arms
        self.strategy = strategy
        self.data_loader = data_loader
        self.rng = np.random.default_rng(config.random_seed)

        # Initialize History Tracker (Telemetry)
        self.history = RegretTelemetry(arm_names=list(arms.keys()))

    def train_arms(self) -> None:
        """Train all arms efficiently using Hybrid Streaming/Sampling.

        Leverages the DataLoader's `stream_training_data` to:
        1. Feed Streaming Arms (Popularity) chunk-by-chunk.
        2. Auto-collect a Reservoir Sample for Batch Arms (Linear/MLP).
        """
        print(f"Starting training phase (Chunks: {self.data_loader.chunk_count})...")
        start_time = time.time()

        # 1. Identify Arm Types
        # Streaming arms must implement `_process_chunk`
        streaming_arms = [arm for arm in self.arms.values() if hasattr(arm, '_process_chunk')]

        # Batch arms need a dataframe, but exclude Random arm
        batch_arms = [
            arm for arm in self.arms.values()
            if not hasattr(arm, '_process_chunk') and arm.name != 'random'
        ]

        # Reset streaming arms (clear old counters)
        for arm in streaming_arms:
            if isinstance(arm, PopularityArm):
                arm.ctr_table = defaultdict(lambda: [0, 0])

        # 2. Initialize Stream from Loader
        # We target ~500k items for Linear/MLP training to prevent RAM explosion.
        # 500k items / 30 items per request = ~16,667 requests.
        target_sample_requests = 16667

        chunk_iter, sampler = self.data_loader.stream_training_data(
            sample_size=target_sample_requests,
            seed=self.config.random_seed
        )

        # 3. Consume Stream
        total_streamed_records = 0

        for chunk_df in chunk_iter:
            total_streamed_records += len(chunk_df)

            # A. Update Streaming Arms (Popularity)
            # The sampler updates itself automatically inside the loader's iterator
            for arm in streaming_arms:
                arm._process_chunk(chunk_df)

        # 4. Finalize Streaming Arms
        for arm in streaming_arms:
            arm._is_trained = True
            print(f"  [Stream] {arm.name} trained on {total_streamed_records} records.")

        # 5. Train Batch Arms (Linear, MLP, KNN)
        if batch_arms:
            # Retrieve the uniform random sample collected by the loader
            train_sample = sampler.get_sample()

            print(f"  [Batch] Training {len(batch_arms)} arms on reservoir of {len(train_sample)} records...")
            for arm in batch_arms:
                print(f"    Training {arm.name}...")
                arm.train(train_sample)

        print(f"Training complete in {time.time() - start_time:.1f}s")

    def _get_best_arm(
        self, record: dict, rankings: Dict[str, np.ndarray]
    ) -> Optional[str]:
        """Determine the Local Best Arm for a specific record, is the arm that placed a relevant item closest to the top (Rank 0).
        """
        labels = np.array(record["labels"])
        clicked_indices = set(np.where(labels == 1.0)[0])

        if not clicked_indices:
            return None  # User clicked nothing, no winner possible

        best_arm = None
        best_rank = float("inf")

        for arm_name, ranking in rankings.items():
            # Find the first clicked item in this arm's list
            for rank, item_idx in enumerate(ranking):
                if item_idx in clicked_indices:
                    # If this is better than the best seen so far, record it
                    if rank < best_rank:
                        best_rank = rank
                        best_arm = arm_name
                    # Stop checking this arm (we only care about its best relevant item)
                    break

        return best_arm

    def run(self, max_iterations: Optional[int] = None) -> RegretTelemetry:
        """Run the MAB simulation on test data."""
        print("Starting Simulation Loop...")
        iteration = 0

        # Use iter_records to stream test data without loading it all into RAM
        for record in self.data_loader.iter_records("test"):
            if max_iterations is not None and iteration >= max_iterations:
                break

            # 1. POLICY STEP: Arms generate rankings
            # (Fast because arms are vectorized)
            rankings = {name: arm.rank(record) for name, arm in self.arms.items()}

            # 2. STRATEGY STEP: Bandit selects arms
            active_arms = self.strategy.select_arms()

            # 3. MONTE CARLO STEP: Sample a slate
            slate, attribution = sample_slate(
                rankings,
                active_arms,
                slate_size=self.config.slate_size,
                rng=self.rng,
            )

            # 4. ENVIRONMENT STEP: Generate stochastic reward
            labels = np.array(record["labels"])
            # Note: We use 'navigational' model for the Bandit update
            winner_info = compute_reward(slate, attribution, labels, click_model="navigational")

            if winner_info:
                winner_name, _ = winner_info
            else:
                winner_name = None

            # 5. UPDATE STEP: Strategy learns
            self.strategy.update(winner_name, active_arms)

            # 6. TELEMETRY STEP: Record history
            best_arm = self._get_best_arm(record, rankings)
            self.history.record_iteration(active_arms, winner_name, best_arm)

            iteration += 1
            if iteration % 1000 == 0:
                stats = self.strategy.get_statistics()
                # Safe access to 'set_E' for printing (some strategies might not have it)
                debug_set = stats.get('set_E', [])
                print(
                    f"  Iter {iteration}: "
                    f"Loss={self.history.total_regret:.0f}, "
                    f"Clicks={sum(self.history.click_history)}, "
                    f"Winners={debug_set}"
                )

        print(f"Simulation complete: {iteration} iterations processed.")
        return self.history

    def get_results(self) -> dict:
        """Return comprehensive experiment results package."""
        num_iter = self.history.num_iterations
        total_regret = self.history.total_regret
        return {
            "config": self.config.__dict__,
            "metrics": {
                "iterations": num_iter,
                "total_iterations": num_iter,
                "total_regret": total_regret,
                "regret_rate": total_regret / num_iter if num_iter > 0 else 0.0,
                "total_clicks": sum(self.history.click_history),
                "global_ctr": sum(self.history.click_history) / max(1, num_iter)
            },
            "selection_rates": self.history.get_arm_selection_rates(),
            "win_rates": self.history.get_arm_win_rates(),
            "strategy": self.strategy.get_statistics(),
            "strategy_stats": self.strategy.get_statistics(),
        }
