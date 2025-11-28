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


def list_available_arms() -> list:
    return list(ARM_REGISTRY.keys())
