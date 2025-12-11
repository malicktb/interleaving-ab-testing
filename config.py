"""Experiment configuration and click model scenarios.

This module contains:
- ExperimentConfig: Dataclass with all experiment parameters
- SCENARIOS: Click model configurations (standard, cascade, noisy)
- get_scenario(): Lookup function for scenarios
"""

from dataclasses import dataclass, field
from typing import Dict, Any


# =============================================================================
# Click Model Scenarios
# =============================================================================

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "standard": {
        "click_model_type": "pbm",
        "params": {},
        "description": "Standard Position-Based Model (Baseline)",
    },
    "cascade": {
        "click_model_type": "cascade",
        "params": {
            "relevance_threshold": 3,
            "max_depth": 10
        },
        "description": "Perfect Cascade: User clicks ONLY the first relevant item (Sparse Feedback)",
    },
    "noisy": {
        "click_model_type": "noisy",
        "params": {
            "relevance_threshold": 3,
            "noise_prob": 0.1,
            "false_negative_rate": 0.1,
            "max_depth": 10
        },
        "description": "Noisy User: 10% chance of random clicks (Robustness Test)",
    },
}


def get_scenario(name: str) -> Dict[str, Any]:
    """Get scenario configuration by name.

    Args:
        name: Scenario name (standard, cascade, noisy).

    Returns:
        Dict with click_model_type, params, and description.

    Raises:
        KeyError: If scenario name is unknown.
    """
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for H-MDB experiments.

    Groups related parameters:
    - Basic: rounds, seed, arms
    - Data: path, scenario, training records
    - Policy: alpha, beta, grace period
    - Arms: PCA, LinUCB, XGBoost, LambdaMART settings
    - Multileaving: scheme, attribution
    - Statistics: tracker type, discount
    - Clustering: HDBSCAN parameters
    - Hierarchical: level1 rounds
    - Cache: score pre-computation settings
    """

    # Basic experiment settings
    n_rounds: int = 10000
    random_seed: int = 42
    arm_pool_list: list = field(default_factory=lambda: ["random", "single_feature", "xgboost", "linucb"])
    arm_pool_size: str = "small"  # small, medium, full
    arm_subset: int = None  # Select K arms from full pool for scalability experiments

    # Data settings
    data_path: str = "data/processed/yahoo_parquet/set1"
    scenario: str = "standard"
    max_train_records: int = 20000

    # Policy settings
    strategy_alpha: float = 0.51
    strategy_beta: float = 1.0
    slate_size: int = 10
    grace_period: int = 500

    # Arm-specific settings
    pca_dim: int = 20
    linucb_alpha: float = 1.0
    xgb_train_fraction: float = 0.1
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    bm25_feature_idx: int = 0
    epsilon_greedy_epsilon: float = 0.1
    lambdamart_n_estimators: int = 100
    lambdamart_max_depth: int = 6
    lambdamart_train_fraction: float = 1.0

    # MultiRUCB configuration
    multi_rucb_m: int = None  # Comparison set size (None = all arms)

    # Multileaving configuration
    multileaving_scheme: str = "team_draft"  # team_draft

    # Attribution scheme configuration
    attribution_scheme: str = "team_draft_legacy"

    # Statistics tracker configuration
    statistics_tracker: str = "cumulative"  # cumulative, discounted
    discount_factor: float = 0.995  # For discounted tracker (γ decay)

    # Clustering configuration (for hierarchical evaluation)
    cluster_min_size: int = 5  # HDBSCAN min_cluster_size
    cluster_sample_queries: int = 1000  # Queries for Jaccard similarity
    cluster_top_k: int = 10  # Top-k docs for similarity computation

    # Hierarchical evaluation configuration (H-MDB-KT)
    hierarchical: bool = False  # Enable two-level hierarchical evaluation
    level1_rounds: int = 2000  # Rounds before Level 1 → Level 2 transition
    clustering_type: str = "output_based"  # output_based or random (RQ3 ablation)
    warm_start: bool = True  # Knowledge Transfer at level transition

    # Score cache configuration (for K=100 scalability)
    use_score_cache: bool = True  # Enable score pre-computation cache
    cache_arm_threshold: int = 10  # Only use cache if >= this many arms
