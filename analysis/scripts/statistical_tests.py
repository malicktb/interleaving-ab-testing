"""Statistical testing utilities for H-MDB experiment analysis.

Provides bootstrap confidence intervals and multiple comparison corrections
for rigorous statistical analysis
"""

from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations.
        statistic: Function to compute statistic (default: mean).
        n_bootstrap: Number of bootstrap samples (default 1000).
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)

    if n == 0:
        return 0.0, 0.0, 0.0

    # Point estimate
    point_estimate = statistic(data)

    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(sample)

    # Percentile confidence interval
    alpha = 1 - ci
    lower = np.percentile(bootstrap_stats, 100 * (alpha / 2))
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(point_estimate), float(lower), float(upper)


def bootstrap_difference(
    data_a: np.ndarray,
    data_b: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence interval for difference between two samples.

    Useful for comparing H-MDB vs flat MDB performance.

    Args:
        data_a: First sample (e.g., H-MDB regrets).
        data_b: Second sample (e.g., flat MDB regrets).
        statistic: Function to compute statistic (default: mean).
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level.
        seed: Random seed.

    Returns:
        Dict with difference estimate, CI bounds, and significance.
    """
    rng = np.random.default_rng(seed)
    data_a = np.asarray(data_a)
    data_b = np.asarray(data_b)
    n_a, n_b = len(data_a), len(data_b)

    if n_a == 0 or n_b == 0:
        return {
            "difference": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "significant": False,
        }

    # Point estimate
    point_diff = statistic(data_a) - statistic(data_b)

    # Bootstrap resampling
    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample_a = rng.choice(data_a, size=n_a, replace=True)
        sample_b = rng.choice(data_b, size=n_b, replace=True)
        bootstrap_diffs[i] = statistic(sample_a) - statistic(sample_b)

    # Percentile CI
    alpha = 1 - ci
    lower = np.percentile(bootstrap_diffs, 100 * (alpha / 2))
    upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    # Significance: CI doesn't include 0
    significant = (lower > 0) or (upper < 0)

    return {
        "difference": float(point_diff),
        "lower": float(lower),
        "upper": float(upper),
        "significant": significant,
    }


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Apply Bonferroni correction for multiple comparisons.

    The Bonferroni correction adjusts the significance threshold to
    alpha / m where m is the number of comparisons.

    Args:
        p_values: List of p-values from individual tests.
        alpha: Family-wise error rate (default 0.05).

    Returns:
        Dict with corrected threshold, significant indices, and details.
    """
    m = len(p_values)
    if m == 0:
        return {
            "corrected_alpha": alpha,
            "n_comparisons": 0,
            "n_significant": 0,
            "significant_indices": [],
            "details": [],
        }

    corrected_alpha = alpha / m
    significant = [i for i, p in enumerate(p_values) if p < corrected_alpha]

    details = []
    for i, p in enumerate(p_values):
        details.append({
            "index": i,
            "p_value": p,
            "significant_raw": p < alpha,
            "significant_corrected": p < corrected_alpha,
        })

    return {
        "corrected_alpha": corrected_alpha,
        "n_comparisons": m,
        "n_significant": len(significant),
        "significant_indices": significant,
        "details": details,
    }


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Apply Holm-Bonferroni step-down correction.

    More powerful than Bonferroni while still controlling FWER.
    Sorts p-values and uses decreasing thresholds.

    Args:
        p_values: List of p-values from individual tests.
        alpha: Family-wise error rate (default 0.05).

    Returns:
        Dict with correction details and significant comparisons.
    """
    m = len(p_values)
    if m == 0:
        return {
            "n_comparisons": 0,
            "n_significant": 0,
            "significant_indices": [],
            "details": [],
        }

    # Sort indices by p-value
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Holm thresholds: alpha / (m - i) for i = 0, 1, ...
    thresholds = alpha / (m - np.arange(m))

    # Find first non-significant
    significant_mask = sorted_p < thresholds
    if np.all(significant_mask):
        k_reject = m
    elif np.any(significant_mask):
        # Find first False (all before are True)
        first_fail = np.argmin(significant_mask)
        k_reject = first_fail
    else:
        k_reject = 0

    # Original indices that are significant
    significant_indices = list(sorted_indices[:k_reject])

    details = []
    for rank, orig_idx in enumerate(sorted_indices):
        details.append({
            "original_index": int(orig_idx),
            "rank": rank + 1,
            "p_value": float(sorted_p[rank]),
            "threshold": float(thresholds[rank]),
            "significant": rank < k_reject,
        })

    return {
        "n_comparisons": m,
        "n_significant": k_reject,
        "significant_indices": [int(x) for x in significant_indices],
        "details": details,
    }


def paired_permutation_test(
    data_a: np.ndarray,
    data_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Non-parametric paired permutation test.

    Tests whether two paired samples have the same distribution.
    Useful for comparing methods on the same queries.

    Args:
        data_a: First sample (paired with data_b).
        data_b: Second sample (same length as data_a).
        n_permutations: Number of permutations (default 10000).
        seed: Random seed.

    Returns:
        Dict with observed difference, p-value (two-tailed), and effect size.
    """
    rng = np.random.default_rng(seed)
    data_a = np.asarray(data_a)
    data_b = np.asarray(data_b)

    if len(data_a) != len(data_b):
        raise ValueError("Paired test requires equal-length arrays")

    n = len(data_a)
    if n == 0:
        return {"observed_diff": 0.0, "p_value": 1.0, "effect_size": 0.0}

    differences = data_a - data_b
    observed_diff = np.mean(differences)

    # Permutation distribution (randomly flip signs)
    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        perm_diffs[i] = np.mean(differences * signs)

    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    # Effect size (Cohen's d)
    std_diff = np.std(differences)
    effect_size = observed_diff / std_diff if std_diff > 0 else 0.0

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
    }


def compute_sample_efficiency(
    success_rounds: List[int],
    total_rounds: int,
    success_threshold: float = 0.95,
) -> Dict[str, Any]:
    """Compute sample efficiency metrics.

    Sample complexity is rounds needed for 95% success rate.

    Args:
        success_rounds: List of round numbers where success occurred.
        total_rounds: Total rounds in experiment.
        success_threshold: Target success rate (default 0.95).

    Returns:
        Dict with sample complexity and efficiency metrics.
    """
    if not success_rounds:
        return {
            "sample_complexity": None,
            "final_success_rate": 0.0,
            "convergence_round": None,
        }

    # Convert to cumulative success rate
    success_array = np.zeros(total_rounds)
    for r in success_rounds:
        if 0 <= r < total_rounds:
            success_array[r] = 1

    cumsum = np.cumsum(success_array)
    rounds = np.arange(1, total_rounds + 1)
    success_rates = cumsum / rounds

    # Find first round where success_rate >= threshold
    above_threshold = np.where(success_rates >= success_threshold)[0]
    sample_complexity = int(above_threshold[0]) + 1 if len(above_threshold) > 0 else None

    return {
        "sample_complexity": sample_complexity,
        "final_success_rate": float(success_rates[-1]) if total_rounds > 0 else 0.0,
        "convergence_round": sample_complexity,
    }


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    data = np.random.normal(10, 2, size=100)

    point, lower, upper = bootstrap_confidence_interval(data)
    print(f"Bootstrap CI for mean: {point:.3f} ({lower:.3f}, {upper:.3f})")

    data_a = np.random.normal(10, 2, size=50)
    data_b = np.random.normal(9, 2, size=50)
    diff_result = bootstrap_difference(data_a, data_b)
    print(f"Difference: {diff_result['difference']:.3f}, significant: {diff_result['significant']}")

    p_values = [0.01, 0.03, 0.05, 0.08, 0.15]
    bonf = bonferroni_correction(p_values)
    print(f"Bonferroni: {bonf['n_significant']} significant at corrected alpha={bonf['corrected_alpha']:.4f}")

    holm = holm_bonferroni_correction(p_values)
    print(f"Holm-Bonferroni: {holm['n_significant']} significant")
