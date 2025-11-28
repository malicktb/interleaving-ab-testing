from typing import List, Dict, Any
import numpy as np
from .base import ClickSimulator


class CascadeModel(ClickSimulator):

    def __init__(self, relevance_threshold: int = 3, max_depth: int = 10):
        self.relevance_threshold = relevance_threshold
        self.max_depth = max_depth

    def simulate(
        self,
        slate: List[int],
        relevance: np.ndarray,
        rng: np.random.Generator,
    ) -> List[int]:
        for pos, item_idx in enumerate(slate):
            if pos >= self.max_depth:
                break

            if item_idx < len(relevance):
                rel_grade = int(relevance[item_idx])
                if rel_grade >= self.relevance_threshold:
                    return [pos]

        return []

    def get_click_probability(self, position: int, relevance: int) -> float:
        if position >= self.max_depth:
            return 0.0
        return 1.0 if relevance >= self.relevance_threshold else 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": "CascadeModel",
            "relevance_threshold": self.relevance_threshold,
            "max_depth": self.max_depth,
        }
