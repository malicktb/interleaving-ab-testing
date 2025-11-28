from collections import defaultdict
from typing import List, Dict, Any, Optional, Set
import numpy as np


def compute_dcg(relevance: np.ndarray, k: int = 10) -> float:
    relevance = np.asarray(relevance)[:k]

    if relevance.size == 0:
        return 0.0

    positions = np.arange(1, len(relevance) + 1)

    return float(np.sum(relevance / np.log2(positions + 1)))


def compute_ndcg(
    relevance: np.ndarray,
    ideal_relevance: Optional[np.ndarray] = None,
    k: int = 10
) -> float:
    dcg = compute_dcg(relevance, k)

    if ideal_relevance is None:
        ideal_relevance = np.sort(relevance)[::-1]

    ideal_dcg = compute_dcg(ideal_relevance, k)

    if ideal_dcg == 0:
        return 1.0

    return dcg / ideal_dcg


class RegretTelemetry:

    def __init__(self, arm_names: List[str]):
        self.arm_names = arm_names
        self.num_iterations = 0

        self.selection_history: List[List[str]] = []
        self.winner_history: List[Optional[str]] = []
        self.click_history: List[int] = []
        self.ndcg_regret_history: List[float] = []
        self.cumulative_ndcg_regret_history: List[float] = []

        self.total_ndcg_regret = 0.0
        self.arm_selection_counts: Dict[str, int] = defaultdict(int)
        self.arm_win_counts: Dict[str, int] = defaultdict(int)

    def record_iteration(
        self,
        selected_arms: List[str],
        winner: Optional[str]
    ) -> None:
        self.num_iterations += 1

        self.selection_history.append(selected_arms)
        for arm in selected_arms:
            self.arm_selection_counts[arm] += 1

        self.winner_history.append(winner)
        has_click = winner is not None
        self.click_history.append(1 if has_click else 0)

        if has_click:
            self.arm_win_counts[winner] += 1

    def record_ndcg_regret(self, optimal_ndcg: float, shown_ndcg: float) -> None:
        ndcg_regret = optimal_ndcg - shown_ndcg

        ndcg_regret = max(0.0, ndcg_regret)

        self.ndcg_regret_history.append(ndcg_regret)
        self.total_ndcg_regret += ndcg_regret
        self.cumulative_ndcg_regret_history.append(self.total_ndcg_regret)

    def get_arm_selection_rates(self) -> Dict[str, float]:
        if self.num_iterations == 0:
            return {}
        return {
            arm: count / self.num_iterations
            for arm, count in self.arm_selection_counts.items()
        }

    def get_arm_win_rates(self) -> Dict[str, float]:
        total_clicks = sum(self.click_history)
        if total_clicks == 0:
            return {}
        return {
            arm: count / total_clicks
            for arm, count in self.arm_win_counts.items()
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "iterations": self.num_iterations,
            "total_ndcg_regret": self.total_ndcg_regret,
            "avg_ndcg_regret": (
                self.total_ndcg_regret / self.num_iterations
                if self.num_iterations > 0 else 0.0
            ),
            "total_clicks": sum(self.click_history),
            "click_rate": (
                sum(self.click_history) / self.num_iterations
                if self.num_iterations > 0 else 0.0
            ),
        }
