SCENARIOS = {
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


def get_scenario(name):
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]
