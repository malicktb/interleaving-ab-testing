from typing import List, Dict, Any
import numpy as np
from .base import ClickSimulator


class NoisyUserModel(ClickSimulator):

    def __init__(
        self,
        relevance_threshold: int = 3,
        noise_prob: float = 0.1,
        false_negative_rate: float = 0.1,
        max_depth: int = 10,
    ):
        self.relevance_threshold = relevance_threshold
        self.noise_prob = noise_prob
        self.false_negative_rate = false_negative_rate
        self.max_depth = max_depth

    def simulate(
        self,
        slate: List[int],
        relevance: np.ndarray,
        rng: np.random.Generator,
    ) -> List[int]:
        effective_depth = min(len(slate), self.max_depth)
        if effective_depth == 0:
            return []

        if rng.random() < self.noise_prob:
            random_pos = rng.integers(0, effective_depth)
            return [random_pos]

        for pos, item_idx in enumerate(slate):
            if pos >= self.max_depth:
                break

            if item_idx < len(relevance):
                rel_grade = int(relevance[item_idx])

                if rel_grade >= self.relevance_threshold:
                    if rng.random() < self.false_negative_rate:
                        continue
                    return [pos]

        return []

    def get_click_probability(self, position: int, relevance: int) -> float:
        if position >= self.max_depth:
            return 0.0

        cascade_prob = 1.0 - self.false_negative_rate if relevance >= self.relevance_threshold else 0.0
        noise_contrib = self.noise_prob / self.max_depth

        return (1 - self.noise_prob) * cascade_prob + noise_contrib

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": "NoisyUserModel",
            "noise_prob": self.noise_prob,
            "false_negative_rate": self.false_negative_rate,
        }
