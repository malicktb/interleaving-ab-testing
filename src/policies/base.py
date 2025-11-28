from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):

    def __init__(self, arm_names):
        self.arm_names = arm_names
        self.K = len(arm_names)
        self.t = 0
        self.W = np.zeros((self.K, self.K), dtype=float)
        self.N = np.zeros((self.K, self.K), dtype=float)

    def _name_to_idx(self, name):
        return self.arm_names.index(name)

    def _idx_to_name(self, idx):
        return self.arm_names[idx]

    @abstractmethod
    def select_arms(self):
        pass

    def update(self, winner, participants):
        self.t += 1

        if winner is None:
            return

        winner_idx = self._name_to_idx(winner)

        for name in participants:
            if name != winner:
                loser_idx = self._name_to_idx(name)
                self.W[winner_idx, loser_idx] += 1
                self.N[winner_idx, loser_idx] += 1
                self.N[loser_idx, winner_idx] += 1

    def get_win_rate(self, i, j):
        if self.N[i, j] == 0:
            return 0.5
        return self.W[i, j] / self.N[i, j]

    def get_statistics(self):
        return {
            "t": self.t,
            "W": self.W.tolist(),
            "N": self.N.tolist(),
        }
