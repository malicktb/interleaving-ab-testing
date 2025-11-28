from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np


class ClickSimulator(ABC):

    @abstractmethod
    def simulate(
        self,
        slate: List[int],
        relevance: np.ndarray,
        rng: np.random.Generator,
    ) -> List[int]:
        pass

    @abstractmethod
    def get_click_probability(self, position: int, relevance: int) -> float:
        pass

    def get_first_click(
        self,
        slate: List[int],
        relevance: np.ndarray,
        rng: np.random.Generator,
    ) -> Optional[int]:
        clicks = self.simulate(slate, relevance, rng)
        return clicks[0] if clicks else None

    def get_stats(self) -> Dict[str, Any]:
        return {"model": self.__class__.__name__}
