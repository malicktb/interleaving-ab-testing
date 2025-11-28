from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:

    n_rounds: int = 10000
    random_seed: int = 42
    arm_pool_list: list = field(default_factory=lambda: ["random", "single_feature", "xgboost", "linucb"])

    data_path: str = "data/processed/yahoo_parquet/set1"
    scenario: str = "standard"
    max_train_records: int = 20000

    strategy_alpha: float = 0.51
    strategy_beta: float = 1.0
    slate_size: int = 10
    grace_period: int = 500

    pca_dim: int = 20
    linucb_alpha: float = 1.0
    xgb_train_fraction: float = 0.1
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6

    def validate(self):
        if self.strategy_alpha <= 0.5:
            raise ValueError("strategy_alpha must be > 0.5")
        if self.grace_period < 0:
            raise ValueError("grace_period must be >= 0")
