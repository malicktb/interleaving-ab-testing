"""Analyze H-MDB-KT Modal evaluation results.

Generates summary statistics for Research Questions:
- RQ1: Inference Efficiency (H-MDB-KT vs Flat MDB)
- RQ2: Knowledge Transfer (Warm Start effect)
- RQ3: Clustering Validity (Output-Based vs Random)
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def analyze_results(results_dir: str = "modal_results/logs"):
    """Analyze all experiment results and generate summary."""
    results_path = Path(results_dir)
    results = defaultdict(list)

    for f in results_path.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)

        # Extract config info from filename
        # Pattern: phase1_1.1_HMDB_KT_seed-0.json or phase2_2.1_K100_Flat_K100_seed-0.json
        stem = f.stem
        parts = stem.split("_")
        phase = parts[0]  # phase1, phase2, phase3

        # Find seed part (last element like "seed-0")
        seed_part = parts[-1]
        seed = int(seed_part.replace("seed-", ""))

        # Extract config_id and name
        config_id = parts[1]

        # For phase2, the name pattern is like "K100_Flat_K100" or "K100_HMDB_K100"
        # For phase1/3, the name pattern is like "HMDB_KT", "Flat_MDB", "Random_KT", "Cascade", "Noisy"
        name_parts = parts[2:-1]  # Everything between config_id and seed

        # Determine the canonical name based on the content
        if phase == "phase2":
            # Phase 2 names: extract Flat vs HMDB and K value
            name_str = "_".join(name_parts)
            if "Flat" in name_str:
                # Extract K value (e.g., K100)
                k_val = name_parts[0] if name_parts[0].startswith("K") else name_parts[-1]
                name = f"Flat_{k_val}"
            elif "HMDB" in name_str:
                k_val = name_parts[0] if name_parts[0].startswith("K") else name_parts[-1]
                name = f"HMDB_{k_val}"
            else:
                name = name_str
        else:
            name = "_".join(name_parts)

        # Extract key metrics from correct locations
        config = data.get("config", {})
        metrics = data.get("metrics", {})
        logical_cost = data.get("logical_cost", {})

        # Key metrics
        total_regret = metrics.get("total_ndcg_regret", 0)
        avg_regret = metrics.get("avg_ndcg_regret", 0)
        total_inferences = logical_cost.get("total_evaluations", 0)
        avg_arms_per_query = logical_cost.get("avg_arms_per_query", 0)
        runtime = data.get("runtime", 0)
        rounds = config.get("n_rounds", 0)
        hierarchical = config.get("hierarchical", False)

        results[name].append({
            "phase": phase,
            "config_id": config_id,
            "seed": seed,
            "total_regret": total_regret,
            "avg_regret": avg_regret,
            "total_inferences": total_inferences,
            "avg_arms_per_query": avg_arms_per_query,
            "runtime": runtime,
            "rounds": rounds,
            "hierarchical": hierarchical,
        })

    print("=" * 70)
    print("H-MDB-KT Evaluation Results Summary")
    print("=" * 70)
    print(f"Total experiments analyzed: {sum(len(v) for v in results.values())}")
    print()

    # Group by phase for reporting
    phase1_configs = ["HMDB_KT", "Flat_MDB", "Random_KT"]
    phase2_configs = [k for k in results.keys() if k.startswith("Flat_K") or k.startswith("HMDB_K")]
    phase3_configs = ["Cascade", "Noisy"]

    print("=" * 70)
    print("PHASE 1: Core Benchmark (RQ1, RQ2, RQ3)")
    print("=" * 70)
    print(f"{'Config':<15} {'Total Regret':<20} {'Avg Arms/Query':<18} {'Runtime (s)':<12}")
    print("-" * 70)

    for name in phase1_configs:
        if name in results:
            regrets = [r["total_regret"] for r in results[name]]
            arms_per_q = [r["avg_arms_per_query"] for r in results[name]]
            runtimes = [r["runtime"] for r in results[name]]
            print(f"{name:<15} {np.mean(regrets):>8.1f} +/- {np.std(regrets):<6.1f}  {np.mean(arms_per_q):<18.2f} {np.mean(runtimes):.1f}")

    print()
    print("=" * 70)
    print("RQ1 (Inference Efficiency): H-MDB-KT vs Flat MDB")
    print("=" * 70)
    if "HMDB_KT" in results and "Flat_MDB" in results:
        hmdb_regret = np.mean([r["total_regret"] for r in results["HMDB_KT"]])
        flat_regret = np.mean([r["total_regret"] for r in results["Flat_MDB"]])
        hmdb_infer = np.mean([r["total_inferences"] for r in results["HMDB_KT"]])
        flat_infer = np.mean([r["total_inferences"] for r in results["Flat_MDB"]])
        hmdb_arms = np.mean([r["avg_arms_per_query"] for r in results["HMDB_KT"]])
        flat_arms = np.mean([r["avg_arms_per_query"] for r in results["Flat_MDB"]])

        print(f"  H-MDB-KT: regret={hmdb_regret:.1f}, inferences={hmdb_infer:.0f}, arms/query={hmdb_arms:.2f}")
        print(f"  Flat MDB: regret={flat_regret:.1f}, inferences={flat_infer:.0f}, arms/query={flat_arms:.2f}")

        if flat_infer > 0:
            cost_reduction = (flat_infer - hmdb_infer) / flat_infer * 100
            print(f"  COST REDUCTION: {cost_reduction:.1f}% fewer inferences")

        if flat_regret > 0:
            regret_diff = ((flat_regret - hmdb_regret) / flat_regret * 100)
            print(f"  REGRET DIFFERENCE: {regret_diff:.1f}%")

    print()
    print("=" * 70)
    print("RQ3 (Clustering Validity): Output-Based vs Random")
    print("=" * 70)
    if "HMDB_KT" in results and "Random_KT" in results:
        hmdb_regret = np.mean([r["total_regret"] for r in results["HMDB_KT"]])
        random_regret = np.mean([r["total_regret"] for r in results["Random_KT"]])
        hmdb_std = np.std([r["total_regret"] for r in results["HMDB_KT"]])
        random_std = np.std([r["total_regret"] for r in results["Random_KT"]])

        print(f"  Output-Based (H-MDB-KT): {hmdb_regret:.1f} +/- {hmdb_std:.1f}")
        print(f"  Random Clustering:       {random_regret:.1f} +/- {random_std:.1f}")
        if random_regret > 0:
            improvement = ((random_regret - hmdb_regret) / random_regret * 100)
            print(f"  OUTPUT-BASED ADVANTAGE: {improvement:.1f}% lower regret")

    print()
    print("=" * 70)
    print("PHASE 2: Scalability Sweep (K = 20, 60, 100)")
    print("=" * 70)
    print(f"{'K arms':<8} {'Flat Regret':<15} {'HMDB Regret':<15} {'Flat Infer':<12} {'HMDB Infer':<12} {'Cost Reduc.':<10}")
    print("-" * 80)

    for K in [20, 60, 100]:
        flat_name = f"Flat_K{K}"
        hmdb_name = f"HMDB_K{K}"

        if flat_name in results and hmdb_name in results:
            flat_regret = np.mean([r["total_regret"] for r in results[flat_name]])
            hmdb_regret = np.mean([r["total_regret"] for r in results[hmdb_name]])
            flat_infer = np.mean([r["total_inferences"] for r in results[flat_name]])
            hmdb_infer = np.mean([r["total_inferences"] for r in results[hmdb_name]])
            cost_red = ((flat_infer - hmdb_infer) / flat_infer * 100) if flat_infer > 0 else 0
            print(f"K={K:<5} {flat_regret:<15.1f} {hmdb_regret:<15.1f} {flat_infer:<12.0f} {hmdb_infer:<12.0f} {cost_red:.1f}%")

    print()
    print("=" * 70)
    print("PHASE 3: Environmental Robustness")
    print("=" * 70)
    print(f"{'Scenario':<15} {'Total Regret':<20} {'Avg Arms/Query':<15}")
    print("-" * 70)

    for name in phase3_configs:
        if name in results:
            regrets = [r["total_regret"] for r in results[name]]
            arms = [r["avg_arms_per_query"] for r in results[name]]
            print(f"{name:<15} {np.mean(regrets):>8.1f} +/- {np.std(regrets):<6.1f}  {np.mean(arms):.2f}")

    # Return raw data for further analysis
    return results


if __name__ == "__main__":
    analyze_results()
