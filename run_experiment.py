import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from config.experiments import ExperimentConfig
from config.scenarios import get_scenario

from src.arms.arm_factory import create_arm
from src.arms.ground_truth import GroundTruthArm

from src.environment.data_loader import DataLoader
from src.environment.simulator import Simulator
from src.environment.click_models.pbm import PositionBasedModel
from src.environment.click_models.cascade import CascadeModel
from src.environment.click_models.noisy import NoisyUserModel

from src.policies.mdb import MDBPolicy
from src.policies.baseline import (
    UniformPolicy,
    SingleArmThompsonSamplingPolicy,
    FixedPolicy
)


def create_arm_pool(config: ExperimentConfig) -> Dict[str, Any]:
    arms = {}
    for arm_type in config.arm_pool_list:
        arms[arm_type] = create_arm(arm_type=arm_type, config=config)
    return arms


def create_click_model(config: ExperimentConfig):
    scenario_def = get_scenario(config.scenario)
    model_type = scenario_def["click_model_type"]
    params = scenario_def["params"].copy()

    print(f"Initializing {model_type} with params: {params}")

    if model_type == "pbm":
        return PositionBasedModel(max_positions=config.slate_size, **params)
    elif model_type == "cascade":
        return CascadeModel(**params)
    elif model_type == "noisy":
        return NoisyUserModel(**params)
    else:
        raise ValueError(f"Unknown click model type: {model_type}")


def create_policy(config: ExperimentConfig, arm_names: List[str]):
    p_type = config.policy.lower()

    if p_type == "mdb":
        return MDBPolicy(
            arm_names=arm_names,
            alpha=config.strategy_alpha,
            beta=config.strategy_beta,
            n_min=config.grace_period,
        )
    elif p_type == "uniform":
        return UniformPolicy(arm_names=arm_names)
    elif p_type == "thompson" or p_type == "ts":
        return SingleArmThompsonSamplingPolicy(arm_names=arm_names, seed=config.random_seed)
    elif p_type.startswith("fixed:"):
        fixed_arm = p_type.split(":")[1]
        return FixedPolicy(arm_names=arm_names, fixed_arm_name=fixed_arm)
    else:
        raise ValueError(f"Unknown policy: {config.policy}")


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    print("=" * 60)
    print(f"EXPERIMENT START: {config.scenario.upper()} SCENARIO")
    print(f"Policy: {config.policy}")
    print(f"Arms: {config.arm_pool_list}")
    print(f"Max Rounds: {config.n_rounds}")
    print("=" * 60)

    start_time = time.time()

    print("\n[1/5] Loading data...")
    data_loader = DataLoader(
        data_dir=config.data_path,
        seed=config.random_seed
    )

    print("\n[2/5] Creating arms...")
    arms = create_arm_pool(config)
    ground_truth = GroundTruthArm(name="ground_truth")

    print("\n[3/5] Creating policy...")
    policy = create_policy(config, list(arms.keys()))

    print("\n[4/5] Creating click model...")
    click_model = create_click_model(config)

    print("\n[5/5] Initializing Simulator...")
    simulator = Simulator(
        arms=arms,
        policy=policy,
        data_loader=data_loader,
        click_model=click_model,
        ground_truth=ground_truth,
        slate_size=config.slate_size,
        random_seed=config.random_seed,
    )

    print("\n--- Phase 1: Offline Training ---")
    simulator.train_arms(max_train_records=config.max_train_records)

    print("\n--- Phase 2: Online Simulation ---")
    simulator.run_episode(
        max_iterations=config.n_rounds,
        log_interval=max(100, config.n_rounds // 10)
    )

    results = simulator.get_results()

    results["config"] = {
        "scenario": config.scenario,
        "policy": config.policy,
        "arms": config.arm_pool_list,
        "n_rounds": config.n_rounds,
        "seed": config.random_seed
    }
    results["runtime"] = time.time() - start_time

    return results


def save_results(results: Dict, output_dir: str = "evaluation_results") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scen = results["config"]["scenario"]
    pol = results["config"]["policy"]

    filename = f"{ts}_{scen}_{pol}.json"
    path = os.path.join(output_dir, filename)

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved results to: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy", type=str, default="mdb")
    parser.add_argument("--scenario", type=str, default="standard")
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--arms", nargs="+", default=["random", "single_feature", "xgboost", "linucb"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()

    config = ExperimentConfig(
        scenario=args.scenario,
        n_rounds=1000 if args.quick else args.rounds,
        arm_pool_list=args.arms,
        random_seed=args.seed,
        max_train_records=1000 if args.quick else 20000
    )

    config.policy = args.policy

    results = run_experiment(config)
    save_results(results)
