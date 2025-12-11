import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from config import ExperimentConfig, get_scenario

from src.arms.factory import create_arm_pool
from src.arms.ground_truth import GroundTruthArm

from simulation.data import DataLoader, ScoreCache, precompute_scores
from simulation import Simulator
from simulation.click_models import PositionBasedModel, CascadeModel, NoisyUserModel

from src.multileaving.attribution import create_attribution_strategy
from src.multileaving.factory import create_multileaver

from src.policies.mdb import MDBPolicy
from src.policies.multi_rucb import MultiRUCBPolicy
from src.policies.hierarchical_mdb import HierarchicalMDBPolicy
from src.policies.baseline import (
    UniformPolicy,
    SingleArmThompsonSamplingPolicy,
    FixedPolicy
)
from src.policies.trackers import create_statistics_tracker
from src.clustering import OutputBasedClusterer, RandomClusterer


def save_trained_arms(arms: Dict[str, Any], output_dir: str) -> str:
    """Save trained arms to disk for reuse.

    Args:
        arms: Dict of arm_name -> trained arm instance.
        output_dir: Directory to save arms.

    Returns:
        Path to saved file.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    arms_file = path / "trained_arms.pkl"
    with open(arms_file, "wb") as f:
        pickle.dump(arms, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Arms] Saved {len(arms)} trained arms to {arms_file}")
    return str(arms_file)


def load_trained_arms(arms_path: str) -> Optional[Dict[str, Any]]:
    """Load pre-trained arms from disk.

    Args:
        arms_path: Path to saved arms file or directory.

    Returns:
        Dict of arm_name -> trained arm instance, or None if not found.
    """
    path = Path(arms_path)

    # Handle both file and directory paths
    if path.is_dir():
        arms_file = path / "trained_arms.pkl"
    else:
        arms_file = path

    if not arms_file.exists():
        print(f"[Arms] No saved arms found at {arms_file}")
        return None

    try:
        with open(arms_file, "rb") as f:
            arms = pickle.load(f)
        print(f"[Arms] Loaded {len(arms)} pre-trained arms from {arms_file}")
        return arms
    except Exception as e:
        print(f"[Arms] Failed to load arms: {e}")
        return None


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

    # Create statistics tracker based on config
    tracker = create_statistics_tracker(
        tracker_type=config.statistics_tracker,
        arm_names=arm_names,
        discount_factor=config.discount_factor,
    )

    if p_type == "mdb":
        return MDBPolicy(
            arm_names=arm_names,
            alpha=config.strategy_alpha,
            beta=config.strategy_beta,
            n_min=config.grace_period,
            statistics_tracker=tracker,
        )
    elif p_type == "multi_rucb":
        return MultiRUCBPolicy(
            arm_names=arm_names,
            alpha=config.strategy_alpha,
            m=config.multi_rucb_m,
            n_min=config.grace_period,
            statistics_tracker=tracker,
            seed=config.random_seed,
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


def create_hierarchical_policy(
    config: ExperimentConfig,
    arms: Dict[str, Any],
    data_loader,
) -> HierarchicalMDBPolicy:
    """Create H-MDB-KT policy with clustering and Knowledge Transfer.

    Args:
        config: Experiment configuration.
        arms: Dict of arm_name -> trained arm instances.
        data_loader: DataLoader for sampling records.

    Returns:
        HierarchicalMDBPolicy configured with clustering result and warm_start.
    """
    clustering_type = getattr(config, 'clustering_type', 'output_based')
    warm_start = getattr(config, 'warm_start', True)

    print(f"\n[H-MDB-KT] Clustering type: {clustering_type}")
    print(f"[H-MDB-KT] Knowledge Transfer (warm_start): {warm_start}")

    # Sample records for clustering (only needed for output_based)
    sample_records = data_loader.sample_train_records(config.cluster_sample_queries)
    validation_records = data_loader.sample_train_records(500)

    # Create clusterer based on type
    if clustering_type == "random":
        # Random clustering for RQ3 ablation
        n_clusters = max(len(arms) // 10, 2)  # ~10 arms per cluster
        clusterer = RandomClusterer(
            n_clusters=n_clusters,
            seed=config.random_seed,
        )
        # Random clustering doesn't need sample records
        clustering_result = clusterer.fit(arms=arms)
        print(f"[H-MDB-KT] Random clustering: {n_clusters} clusters")
    else:
        # Output-based clustering (default)
        clusterer = OutputBasedClusterer(
            min_cluster_size=config.cluster_min_size,
            k=config.cluster_top_k,
            n_sample_queries=config.cluster_sample_queries,
            seed=config.random_seed,
        )
        # Run clustering on trained arms
        clustering_result = clusterer.fit(
            arms=arms,
            sample_records=sample_records,
            validation_records=validation_records,
        )

    print(f"[H-MDB-KT] Clustering complete: {len(clustering_result.clusters)} clusters")
    print(f"[H-MDB-KT] Representatives: {clustering_result.representatives}")
    print(f"[H-MDB-KT] Level 1 arms: {clustering_result.get_level1_arms()}")

    # Create statistics tracker
    tracker = create_statistics_tracker(
        tracker_type=config.statistics_tracker,
        arm_names=list(arms.keys()),
        discount_factor=config.discount_factor,
    )

    # Create hierarchical policy with Knowledge Transfer
    return HierarchicalMDBPolicy(
        arm_names=list(arms.keys()),
        clustering_result=clustering_result,
        level1_rounds=config.level1_rounds,
        alpha=config.strategy_alpha,
        beta=config.strategy_beta,
        n_min=config.grace_period,
        warm_start=warm_start,
        statistics_tracker=tracker,
    )


def run_experiment(
    config: ExperimentConfig,
    load_arms_path: Optional[str] = None,
    save_arms_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 60)
    print(f"EXPERIMENT START: {config.scenario.upper()} SCENARIO")
    print(f"Policy: {config.policy}")
    print(f"Arms: {config.arm_pool_list}")
    print(f"Max Rounds: {config.n_rounds}")
    if config.hierarchical:
        print(f"Mode: HIERARCHICAL (Level 1: {config.level1_rounds} rounds)")
    print("=" * 60)

    start_time = time.time()

    print("\n[1/6] Loading data...")
    data_loader = DataLoader(
        data_dir=config.data_path,
        seed=config.random_seed
    )

    # Try to load pre-trained arms if path provided
    arms = None
    arms_were_loaded = False
    if load_arms_path:
        print(f"\n[2/6] Loading pre-trained arms from {load_arms_path}...")
        arms = load_trained_arms(load_arms_path)
        if arms:
            arms_were_loaded = True
            print(f"Loaded {len(arms)} pre-trained arms (skipping training)")

    # Create new arms if not loaded
    if arms is None:
        print("\n[2/6] Creating arms...")
        arms = create_arm_pool(config, size=config.arm_pool_size)
        print(f"Created {len(arms)} arms (pool size: {config.arm_pool_size})")

    # Subset to K arms if specified (for scalability experiments)
    if config.arm_subset is not None and config.arm_subset < len(arms):
        rng = np.random.default_rng(config.random_seed)
        all_arm_names = list(arms.keys())
        selected_names = rng.choice(all_arm_names, size=config.arm_subset, replace=False)
        arms = {name: arms[name] for name in selected_names}
        print(f"[Subset] Selected {config.arm_subset} arms from pool of {len(all_arm_names)}")

    ground_truth = GroundTruthArm(name="ground_truth")

    print("\n[3/6] Creating click model...")
    click_model = create_click_model(config)

    print("\n[4/6] Creating attribution strategy...")
    attribution_strategy = create_attribution_strategy(
        strategy_type=config.attribution_scheme,
    )
    print(f"Using attribution: {config.attribution_scheme}")

    # For hierarchical mode, train arms BEFORE creating policy (clustering needs trained arms)
    if config.hierarchical:
        if not arms_were_loaded:
            print("\n--- Pre-Phase: Training arms for clustering ---")
            train_records = data_loader.sample_train_records(config.max_train_records)
            for arm_name, arm in arms.items():
                print(f"  Training {arm_name}...")
                arm.train(train_records)

            # Save trained arms if requested
            if save_arms_path:
                save_trained_arms(arms, save_arms_path)
        else:
            print("\n--- Skipping training: Using pre-trained arms ---")

        print("\n[5/6] Creating hierarchical policy with clustering...")
        policy = create_hierarchical_policy(config, arms, data_loader)
    else:
        print("\n[5/6] Creating policy...")
        policy = create_policy(config, list(arms.keys()))

    print("\n[6/6] Initializing Simulator...")
    multileaver = create_multileaver(
        scheme=config.multileaving_scheme,
    )

    simulator = Simulator(
        arms=arms,
        policy=policy,
        data_loader=data_loader,
        click_model=click_model,
        ground_truth=ground_truth,
        slate_size=config.slate_size,
        random_seed=config.random_seed,
        attribution_strategy=attribution_strategy,
        multileaver=multileaver,
        hierarchical=config.hierarchical,
    )

    # Initialize cost calculator for hierarchical mode
    if config.hierarchical and hasattr(policy, 'clustering'):
        clustering_result = policy.clustering
        level1_count = len(clustering_result.get_level1_arms())
        # Level 2 count depends on winning cluster, estimate with average cluster size
        avg_cluster_size = len(arms) // max(len(clustering_result.clusters), 1)
        level2_count = avg_cluster_size
        simulator.initialize_cost_calculator(
            total_arms=len(arms),
            level1_arms=level1_count,
            level2_arms=level2_count,
        )

    # For non-hierarchical mode, train arms in simulator (unless pre-loaded)
    if not config.hierarchical:
        if not arms_were_loaded:
            print("\n--- Phase 1: Offline Training ---")
            simulator.train_arms(max_train_records=config.max_train_records)

            # Save trained arms if requested
            if save_arms_path:
                save_trained_arms(arms, save_arms_path)
        else:
            print("\n--- Skipping Phase 1: Using pre-trained arms ---")
    else:
        print("\n--- Skipping Phase 1: Arms already trained for clustering ---")

    # Build score cache for scalability (only for large arm pools)
    test_records = None
    score_cache = None
    if config.use_score_cache and len(arms) >= config.cache_arm_threshold:
        print("\n--- Building Score Cache ---")
        # Load all test records upfront
        test_records = list(data_loader.iter_test_records())
        if config.n_rounds and len(test_records) > config.n_rounds:
            test_records = test_records[:config.n_rounds]

        # Build score cache for all arms
        score_cache = ScoreCache()
        cache_stats = score_cache.build(
            arms=arms,
            records=test_records,
            exclude_arms=[],
            verbose=True,
        )
        print(f"  Cache memory: {cache_stats['memory_mb']:.1f} MB")

        # Update simulator with cache
        simulator.score_cache = score_cache
    elif config.use_score_cache:
        print(f"\n[Cache] Skipping cache: {len(arms)} arms < threshold {config.cache_arm_threshold}")

    print("\n--- Phase 2: Online Simulation ---")
    simulator.run_episode(
        max_iterations=config.n_rounds,
        log_interval=max(100, config.n_rounds // 10),
        test_records=test_records,
    )

    results = simulator.get_results()

    results["config"] = {
        "scenario": config.scenario,
        "policy": config.policy,
        "arms": config.arm_pool_list,
        "arm_pool_size": config.arm_pool_size,
        "arm_subset": config.arm_subset,
        "n_rounds": config.n_rounds,
        "seed": config.random_seed,
        "attribution_scheme": config.attribution_scheme,
        "statistics_tracker": config.statistics_tracker,
        "discount_factor": config.discount_factor,
        "multileaving_scheme": config.multileaving_scheme,
        "cluster_min_size": config.cluster_min_size,
        "cluster_sample_queries": config.cluster_sample_queries,
        "cluster_top_k": config.cluster_top_k,
        "hierarchical": config.hierarchical,
        "level1_rounds": config.level1_rounds,
        "clustering_type": getattr(config, 'clustering_type', 'output_based'),
        "warm_start": getattr(config, 'warm_start', True),
    }
    results["runtime"] = time.time() - start_time

    return results


def save_results(
    results: Dict,
    output_path: Optional[str] = None,
    output_dir: str = "evaluation_results"
) -> str:
    """Saves results to a specific file or a timestamped file in output_dir.

    Args:
        results: Dict of experiment results.
        output_path: If provided, exact path to write (overrides default naming).
        output_dir: Directory for default timestamped files.

    Returns:
        Path to saved file.
    """
    if output_path:
        # User provided exact path (e.g., via Orchestrator)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Default behavior: Timestamped file in output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        scen = results["config"]["scenario"]
        pol = results["config"]["policy"]
        filename = f"{ts}_{scen}_{pol}.json"
        path = Path(output_dir) / filename

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved results to: {path}")
    return str(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy", type=str, default="mdb")
    parser.add_argument("--scenario", type=str, default="standard")
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--arms", nargs="+", default=["random", "single_feature", "xgboost", "linucb"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--arm-pool",
        type=str,
        default="small",
        choices=["small", "medium", "full"],
        help="Arm pool size: small (uses --arms list), medium (~42 arms), full (~100 arms)",
    )
    parser.add_argument(
        "--arm-subset",
        type=int,
        default=None,
        help="Select K arms from full pool for scalability experiments (e.g., --arm-subset 20 for K=20)",
    )
    parser.add_argument(
        "--multileaving",
        type=str,
        default="team_draft",
        choices=["team_draft"],
        help="Multileaving scheme: team_draft (default)",
    )

    # Attribution arguments
    parser.add_argument(
        "--attribution",
        type=str,
        default="team_draft_legacy",
        choices=["team_draft_legacy"],
        help="Attribution scheme for credit assignment"
    )

    # MultiRUCB specific arguments
    parser.add_argument(
        "--multi-rucb-m",
        type=int,
        default=None,
        help="Comparison set size for MultiRUCB (default: all arms)"
    )

    # Statistics tracker arguments
    parser.add_argument(
        "--tracker",
        type=str,
        default="cumulative",
        choices=["cumulative", "discounted"],
        help="Statistics tracker type (cumulative or discounted)"
    )
    parser.add_argument(
        "--discount-factor",
        type=float,
        default=0.995,
        help="Discount factor γ for discounted tracker (only used with --tracker discounted)"
    )

    # Clustering arguments (for hierarchical evaluation)
    parser.add_argument(
        "--cluster-min-size",
        type=int,
        default=5,
        help="Minimum cluster size for HDBSCAN (default: 5)"
    )
    parser.add_argument(
        "--cluster-sample-queries",
        type=int,
        default=1000,
        help="Number of queries for Jaccard similarity computation (default: 1000)"
    )
    parser.add_argument(
        "--cluster-top-k",
        type=int,
        default=10,
        help="Top-k documents for similarity computation (default: 10)"
    )
    # Hierarchical evaluation arguments (H-MDB-KT)
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Enable two-level hierarchical evaluation (H-MDB-KT)"
    )
    parser.add_argument(
        "--level1-rounds",
        type=int,
        default=2000,
        help="Rounds for Level 1 before transition to Level 2 (default: 2000)"
    )
    parser.add_argument(
        "--clustering",
        type=str,
        default="output_based",
        choices=["output_based", "random"],
        help="Clustering method: output_based (Jaccard similarity) or random (RQ3 ablation)"
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        default=True,
        help="Use Knowledge Transfer at level transition (default: True)"
    )
    parser.add_argument(
        "--no-warm-start",
        dest="warm_start",
        action="store_false",
        help="Disable Knowledge Transfer (cold start ablation)"
    )

    # Score cache arguments (for K=100 scalability)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable score pre-computation cache"
    )
    parser.add_argument(
        "--cache-threshold",
        type=int,
        default=10,
        help="Only use cache if >= this many arms (default: 10)"
    )

    # Model persistence arguments (for repeated experiments)
    parser.add_argument(
        "--save-arms",
        type=str,
        default=None,
        help="Save trained arms to this directory for reuse"
    )
    parser.add_argument(
        "--load-arms",
        type=str,
        default=None,
        help="Load pre-trained arms from this path (skips training)"
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="Specific output file path. If provided, overrides default timestamp naming."
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        scenario=args.scenario,
        n_rounds=1000 if args.quick else args.rounds,
        arm_pool_list=args.arms,
        arm_pool_size="full" if args.arm_subset else args.arm_pool,  # Force full if subsetting
        arm_subset=args.arm_subset,
        random_seed=args.seed,
        max_train_records=1000 if args.quick else 20000,
        attribution_scheme=args.attribution,
        statistics_tracker=args.tracker,
        discount_factor=args.discount_factor,
        multileaving_scheme=args.multileaving,
        multi_rucb_m=args.multi_rucb_m,
        cluster_min_size=args.cluster_min_size,
        cluster_sample_queries=args.cluster_sample_queries,
        cluster_top_k=args.cluster_top_k,
        hierarchical=args.hierarchical,
        level1_rounds=args.level1_rounds,
        clustering_type=args.clustering,
        warm_start=args.warm_start,
        use_score_cache=not args.no_cache,
        cache_arm_threshold=args.cache_threshold,
    )

    config.policy = args.policy

    results = run_experiment(
        config,
        load_arms_path=args.load_arms,
        save_arms_path=args.save_arms,
    )
    save_results(results, output_path=args.logfile)
