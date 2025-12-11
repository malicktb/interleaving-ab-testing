"""Factory functions for creating ranking arms.

Provides:
- ARM_REGISTRY: Mapping of arm type names to classes
- create_arm(): Create single arm instance with config
- create_arm_pool(): Create pools of varying sizes
- generate_xgboost_grid(): XGBoost hyperparameter grid
"""

from typing import Dict, Any

from .base import BaseArm
from .ground_truth import GroundTruthArm
from .random import RandomArm
from .single_feature import SingleFeatureArm
from .linucb import LinUCBArm
from .linear_ts import LinearTSArm
from .xgboost import XGBoostArm


ARM_REGISTRY = {
    "ground_truth": GroundTruthArm,
    "random": RandomArm,
    "single_feature": SingleFeatureArm,
    "linucb": LinUCBArm,
    "linear_ts": LinearTSArm,
    "xgboost": XGBoostArm,
}


def create_arm(arm_type: str, config=None, **kwargs) -> BaseArm:
    """Create a single arm instance.

    Args:
        arm_type: Key from ARM_REGISTRY.
        config: Optional ExperimentConfig for default parameters.
        **kwargs: Override parameters.

    Returns:
        Configured arm instance.
    """
    arm_class = ARM_REGISTRY[arm_type]
    params = {}

    if config is not None:
        if arm_type == "linucb":
            params["alpha"] = config.linucb_alpha
            params["feature_dim"] = config.pca_dim
            params["use_pca"] = True
        elif arm_type == "linear_ts":
            params["feature_dim"] = config.pca_dim
            params["use_pca"] = True
            params["seed"] = config.random_seed
        elif arm_type == "xgboost":
            params["n_estimators"] = config.xgb_n_estimators
            params["max_depth"] = config.xgb_max_depth
            params["train_fraction"] = config.xgb_train_fraction
            params["random_state"] = config.random_seed
        elif arm_type == "random":
            params["seed"] = config.random_seed

    params.update(kwargs)
    return arm_class(name=arm_type, **params)


def generate_xgboost_grid(config, size: str = "medium") -> Dict[str, Any]:
    """Generate XGBoost variants with hyperparameter grid.

    Per main.tex Section 2: "Candidate Pool (K=100): A dense hyperparameter sweep"

    Grid dimensions for "full" pool (K≈120):
    - max_depth: [3, 5, 6, 8, 10] = 5 values
    - n_estimators: [50, 100] = 2 values
    - learning_rate: [0.05, 0.1, 0.2] = 3 values
    - subsample: [0.7, 1.0] = 2 values
    - colsample_bytree: [0.7, 1.0] = 2 values
    Total: 5 × 2 × 3 × 2 × 2 = 120 arms

    Args:
        config: ExperimentConfig with random_seed.
        size: "medium" (~24 arms) or "full" (~120 arms for K=100 experiments).

    Returns:
        Dict of arm_name -> XGBoostArm instance.
    """
    arms = {}

    if size == "medium":
        # Medium grid: ~24 arms (for faster development/testing)
        depths = [3, 6, 10]
        n_estimators_list = [50, 100]
        learning_rates = [0.05, 0.1, 0.2]
        subsamples = [1.0]  # No subsample variation for medium
        colsample_bytrees = [1.0]  # No colsample variation for medium
    else:  # full
        # Full grid: ~120 arms (for K=100 scalability experiments)
        depths = [3, 5, 6, 8, 10]
        n_estimators_list = [50, 100]
        learning_rates = [0.05, 0.1, 0.2]
        subsamples = [0.7, 1.0]
        colsample_bytrees = [0.7, 1.0]

    for depth in depths:
        for n_est in n_estimators_list:
            for lr in learning_rates:
                for subsample in subsamples:
                    for colsample in colsample_bytrees:
                        # Create descriptive name
                        name = f"xgb_d{depth}_n{n_est}_lr{int(lr*100)}"
                        if subsample != 1.0:
                            name += f"_ss{int(subsample*100)}"
                        if colsample != 1.0:
                            name += f"_cs{int(colsample*100)}"

                        arms[name] = XGBoostArm(
                            name=name,
                            max_depth=depth,
                            n_estimators=n_est,
                            learning_rate=lr,
                            subsample=subsample,
                            colsample_bytree=colsample,
                            random_state=config.random_seed,
                        )
    return arms


def create_arm_pool(config, size: str = "small") -> Dict[str, Any]:
    """Create arm pool of specified size.

    Args:
        config: ExperimentConfig.
        size: "small" (uses config.arm_pool_list), "medium" (~22 arms), "full" (~124 arms).

    Returns:
        Dict of arm_name -> arm instance.

    Pool sizes:
        - small: Explicit arm list from config (typically 4 arms)
        - medium: 4 base + 18 XGBoost = 22 arms
        - full: 4 base + 120 XGBoost = 124 arms (for K=100 experiments)
    """
    arms = {}

    if size == "small":
        # Use the explicit arm list from config
        for arm_type in config.arm_pool_list:
            arms[arm_type] = create_arm(arm_type, config)
        return arms

    # For medium/full, build comprehensive pool

    # Base arms (always included)
    base_arms = ["random", "single_feature", "linucb", "linear_ts"]
    for arm_type in base_arms:
        arms[arm_type] = create_arm(arm_type, config)

    # XGBoost grid (main source of arm diversity)
    arms.update(generate_xgboost_grid(config, size))

    return arms
