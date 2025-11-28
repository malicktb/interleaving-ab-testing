import numpy as np
from .base import BasePolicy


class UniformPolicy(BasePolicy):

    def select_arms(self):
        return list(self.arm_names)


class FixedPolicy(BasePolicy):

    def __init__(self, arm_names, fixed_arm_name):
        super().__init__(arm_names)
        if fixed_arm_name not in arm_names:
            raise ValueError(f"Arm '{fixed_arm_name}' not in {arm_names}")
        self.fixed_arm = fixed_arm_name

    def select_arms(self):
        return [self.fixed_arm]


class SingleArmThompsonSamplingPolicy(BasePolicy):

    def __init__(self, arm_names, seed=42):
        super().__init__(arm_names)
        self.rng = np.random.default_rng(seed)
        self.alphas = np.ones(self.K)
        self.betas = np.ones(self.K)

    def select_arms(self):
        samples = self.rng.beta(self.alphas, self.betas)
        best_idx = np.argmax(samples)
        return [self.arm_names[best_idx]]

    def update(self, winner, participants):
        self.t += 1
        played_arm = participants[0]
        played_idx = self.arm_names.index(played_arm)

        if winner == played_arm:
            self.alphas[played_idx] += 1
        else:
            self.betas[played_idx] += 1

    def get_statistics(self):
        stats = super().get_statistics()
        stats["alphas"] = self.alphas.tolist()
        stats["betas"] = self.betas.tolist()
        estimates = self.alphas / (self.alphas + self.betas)
        stats["estimated_rates"] = {
            name: float(estimates[i])
            for i, name in enumerate(self.arm_names)
        }
        return stats
