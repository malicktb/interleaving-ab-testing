"""
This file holds all the settings for the experiment. Changing these values
changes how the Bandit behaves and how much data is used.
"""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configuration for the Multi-Dueling Bandit experiment.

    Attributes:
        alpha: Exploration Knob.
               Higher = The Bandit is more curious/optimistic about uncertain arms.
               Must be > 0.5 to ensure the math works.
        beta:  Parallelism Knob.
               1.0 = Conservative testing. 2.0 = Aggressive parallel testing.
        slate_size: Screen Real Estate.
               How many items to show the user (e.g., Top 10).
        train_ratio: Data Split.
               Percentage of data used to teach the arms before the test starts (0.8 = 80%).
        split_seed: Reproducibility Key.
               Ensures Train/Test splits are identical across different runs.
        num_chunks: Data Limit.
               How many file chunks to load. Set to None to use the full dataset.
        random_seed: Simulation Randomness.
               Seed for the Bandit's coin flips (Team Draft ordering, etc.).
        linear_sample_size: RAM Safety Limit.
               How many user requests to sample for training batch arms (e.g., LinearArm).
               16,667 requests * 30 items ≈ 500,000 rows
    """

    # Bandit Strategy Settings (Brost et al., 2016)
    alpha: float = 0.51
    beta: float = 1.0

    # Environment Settings
    slate_size: int = 10
    train_ratio: float = 0.8
    split_seed: int = 42
    num_chunks: int = None
    random_seed: int = 42
    
    # Memory Safety
    linear_sample_size: int = 16667 

    def __post_init__(self):
        """Sanity checks to prevent invalid configurations."""
        if self.alpha <= 0.5:
            raise ValueError("alpha must be > 0.5")
        if self.beta < 1.0:
            raise ValueError("beta must be >= 1.0")
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if self.slate_size < 1 or self.slate_size > 30:
            raise ValueError("slate_size must be between 1 and 30")