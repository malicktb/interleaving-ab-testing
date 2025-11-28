import time
import numpy as np
from typing import Dict, Any, Optional, List

from ..multileaving.team_draft import interleave
from ..multileaving.attribution import get_click_winner
from ..utils.profiler import Profiler
from ..utils.metrics import RegretTelemetry
from .data_loader import QueryRecord


class Simulator:

    def __init__(
        self,
        arms: Dict[str, Any],
        policy: Any,
        data_loader: Any,
        click_model: Any,
        ground_truth: Any = None,
        slate_size: int = 10,
        random_seed: int = 42,
    ):
        self.arms = arms
        self.policy = policy
        self.data_loader = data_loader
        self.click_model = click_model
        self.ground_truth = ground_truth
        self.slate_size = slate_size
        self.rng = np.random.default_rng(random_seed)

        self.history: Optional[RegretTelemetry] = None
        self._iteration = 0

        self.profiler = Profiler()

    def reset(self):
        self.history = RegretTelemetry(arm_names=list(self.arms.keys()))
        self._iteration = 0
        self.profiler.reset()
        return {"iteration": 0}

    def train_arms(self, max_train_records: Optional[int] = None):
        print("Starting training phase...")
        start_time = time.time()

        if hasattr(self.data_loader, 'sample_train_records'):
            train_records = self.data_loader.sample_train_records(max_train_records)
        else:
            train_records = list(self.data_loader.iter_train_records())
            if max_train_records:
                train_records = train_records[:max_train_records]

        print(f"  Collected {len(train_records)} training records")

        for name, arm in self.arms.items():
            print(f"  Training {name}...")
            arm.train(train_records)

        if self.ground_truth and not self.ground_truth.is_trained:
            self.ground_truth.train(train_records)

        print(f"Training complete in {time.time() - start_time:.1f}s")

    def step(self, record: QueryRecord) -> Dict[str, Any]:
        if self.history is None:
            self.reset()

        self.profiler.increment_query()

        active_arm_names = self.policy.select_arms()

        rankings = {}
        for name in active_arm_names:
            arm = self.arms[name]
            rankings[name] = arm.rank(record)
            self.profiler.increment_inference(1, arm_name=name)

        slate, attribution = interleave(
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

        winner_name = get_click_winner(clicks, slate, attribution)

        self.policy.update(winner_name, active_arm_names)

        if clicks:
            first_click_idx = clicks[0]
            clicked_doc_idx = slate[first_click_idx]
            clicked_features = record.features[clicked_doc_idx]

            for arm in self.arms.values():
                if hasattr(arm, "update"):
                    arm.update(clicked_features, reward=1.0)

        slate_ndcg = 0.0

        self.history.record_iteration(active_arm_names, winner_name)

        if self.ground_truth:
            optimal_ranking = self.ground_truth.rank(record)
            optimal_ndcg = self.ground_truth.compute_ndcg(
                record, optimal_ranking, k=self.slate_size
            )
            slate_ndcg = self.ground_truth.compute_ndcg(
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

    def run_episode(self, max_iterations=None, log_interval=1000):
        print("Starting simulation loop...")
        self.reset()

        for record in self.data_loader.iter_test_records():
            if max_iterations and self._iteration >= max_iterations:
                break

            self.step(record)

            if self._iteration % log_interval == 0:
                self._log_progress()

        print(f"Simulation complete: {self._iteration} iterations.")
        return self.history

    def _log_progress(self):
        stats = self.policy.get_statistics()
        set_e = stats.get("set_E", [])
        in_grace = stats.get("in_grace_period", False)

        regret = self.history.total_ndcg_regret
        avg_ndcg = (self.history.total_ndcg_regret / self._iteration) if self._iteration > 0 else 0

        grace_str = " [GRACE]" if in_grace else ""

        print(
            f"  Iter {self._iteration}: "
            f"CumRegret={regret:.1f}, "
            f"Cost={self.profiler.get_total_cost()}, "
            f"Winners={set_e}{grace_str}"
        )

    def get_results(self):
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

        return results
