import numpy as np
from .base import BaseArm


class RandomArm(BaseArm):

    def __init__(self, name="random", seed=42):
        super().__init__(name)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def train(self, train_records):
        self._is_trained = True

    def rank(self, record):
        indices = np.arange(record.num_items)
        self.rng.shuffle(indices)
        return indices
