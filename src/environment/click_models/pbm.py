from typing import List, Dict, Optional, Any
import numpy as np
from .base import ClickSimulator


class PositionBasedModel(ClickSimulator):

    DEFAULT_EXAMINATION_PROBS = [
        1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10
    ]

    DEFAULT_CLICK_PROBS = {
        0: 0.01,
        1: 0.10,
        2: 0.30,
        3: 0.60,
        4: 0.95,
    }

    def __init__(
        self,
        examination_probs: Optional[List[float]] = None,
        click_probs: Optional[Dict[int, float]] = None,
        max_positions: int = 10,
    ):
        if examination_probs:
            self.examination_probs = list(examination_probs)
        else:
            self.examination_probs = self.DEFAULT_EXAMINATION_PROBS.copy()

        self.click_probs = click_probs or self.DEFAULT_CLICK_PROBS.copy()
        self.max_positions = max_positions

        while len(self.examination_probs) < max_positions:
            last = self.examination_probs[-1]
            self.examination_probs.append(max(0.01, last * 0.9))

    def simulate(
        self,
        slate: List[int],
        relevance: np.ndarray,
        rng: np.random.Generator,
    ) -> List[int]:
        clicks = []

        for pos, item_idx in enumerate(slate):
            if pos >= len(self.examination_probs):
                break

            p_examine = self.examination_probs[pos]

            if item_idx < len(relevance):
                rel_grade = int(relevance[item_idx])
                p_click_given_rel = self.click_probs.get(rel_grade, 0.0)
            else:
                p_click_given_rel = 0.0

            if rng.random() < (p_examine * p_click_given_rel):
                clicks.append(pos)

        return clicks

    def get_click_probability(self, position: int, relevance: int) -> float:
        if position >= len(self.examination_probs):
            return 0.0
        p_examine = self.examination_probs[position]
        p_click_given_rel = self.click_probs.get(relevance, 0.0)
        return p_examine * p_click_given_rel

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": "PositionBasedModel",
            "p_examine_top": self.examination_probs[0],
            "p_click_rel4": self.click_probs.get(4, 0.0),
        }
