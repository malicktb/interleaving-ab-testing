"""
Experiment Results Analysis Script
Analyzes JSON output files from MDB experiments and prints summary statistics.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def load_results(results_dir: str = "evaluation_results") -> List[Dict[str, Any]]:
    """Load all JSON result files from the results directory."""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' not found")
        return results

    for json_file in sorted(results_path.glob("*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            data['_filename'] = json_file.name
            results.append(data)

    return results


def categorize_results(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize results by experiment type."""
    categories = defaultdict(list)

    for r in results:
        config = r.get('config', {})
        scenario = config.get('scenario', 'unknown')
        policy = config.get('policy', 'unknown')
        arms = tuple(sorted(config.get('arms', [])))

        # Categorize by arm pool (use set comparison for order-independence)
        arms_set = set(arms)
        if arms_set == {'linucb', 'linear_ts'}:
            arm_pool = 'learning_only'
        elif 'linear_ts' in arms_set and len(arms) == 5:
            arm_pool = 'full_pool'
        else:
            arm_pool = 'base_pool'

        categories[f"{scenario}_{arm_pool}"].append(r)

    return categories


def print_policy_comparison(results: List[Dict], scenario: str, arm_pool: str):
    """Print comparison table for different policies in same scenario."""
    print(f"\n{'='*80}")
    print(f"POLICY COMPARISON: {scenario.upper()} Scenario ({arm_pool})")
    print(f"{'='*80}")

    # Group by policy
    by_policy = defaultdict(list)
    for r in results:
        policy = r.get('config', {}).get('policy', 'unknown')
        by_policy[policy].append(r)

    # Print header
    print(f"\n{'Policy':<18} {'Total Regret':>12} {'Avg Regret':>12} {'Click Rate':>12} {'Inferences':>12} {'Arms/Query':>12}")
    print("-" * 80)

    rows = []
    for policy, policy_results in sorted(by_policy.items()):
        # Average across runs with same policy
        total_regret = sum(r['metrics']['total_ndcg_regret'] for r in policy_results) / len(policy_results)
        avg_regret = sum(r['metrics']['avg_ndcg_regret'] for r in policy_results) / len(policy_results)
        click_rate = sum(r['metrics']['click_rate'] for r in policy_results) / len(policy_results)
        total_inferences = sum(r['inference_stats']['total_inferences'] for r in policy_results) / len(policy_results)
        avg_arms = sum(r['inference_stats']['avg_arms_per_query'] for r in policy_results) / len(policy_results)

        rows.append((policy, total_regret, avg_regret, click_rate, total_inferences, avg_arms))

    # Sort by total regret
    rows.sort(key=lambda x: x[1])

    for policy, total_regret, avg_regret, click_rate, total_inferences, avg_arms in rows:
        print(f"{policy:<18} {total_regret:>12.1f} {avg_regret:>12.4f} {click_rate:>12.2%} {total_inferences:>12.0f} {avg_arms:>12.2f}")


def print_mdb_convergence(results: List[Dict]):
    """Print MDB convergence analysis."""
    print(f"\n{'='*80}")
    print("MDB CONVERGENCE ANALYSIS")
    print(f"{'='*80}")

    mdb_results = [r for r in results if r.get('config', {}).get('policy') == 'mdb']

    print(f"\n{'Scenario':<12} {'Arm Pool':<15} {'Final Set E':<30} {'Cost':>10}")
    print("-" * 70)

    for r in sorted(mdb_results, key=lambda x: (x['config']['scenario'], str(x['config']['arms']))):
        scenario = r['config']['scenario']
        arms = r['config']['arms']

        if set(arms) == {'linucb', 'linear_ts'}:
            arm_pool = 'learning_only'
        elif 'linear_ts' in arms and len(arms) == 5:
            arm_pool = 'full_pool'
        else:
            arm_pool = 'base_pool'

        set_e = r.get('policy_stats', {}).get('set_E', [])
        cost = r.get('inference_stats', {}).get('total_inferences', 0)

        print(f"{scenario:<12} {arm_pool:<15} {str(set_e):<30} {cost:>10}")


def print_arm_performance(results: List[Dict]):
    """Print arm win rates across experiments."""
    print(f"\n{'='*80}")
    print("ARM WIN RATES (Base Pool, All Scenarios)")
    print(f"{'='*80}")

    # Filter to base pool experiments with mdb or uniform
    base_results = [r for r in results
                   if set(r.get('config', {}).get('arms', [])) == {'random', 'single_feature', 'xgboost', 'linucb'}
                   and r.get('config', {}).get('policy') in ('mdb', 'uniform')]

    # Group by scenario
    by_scenario = defaultdict(list)
    for r in base_results:
        by_scenario[r['config']['scenario']].append(r)

    for scenario in ['standard', 'cascade', 'noisy']:
        if scenario not in by_scenario:
            continue

        print(f"\n{scenario.upper()} Scenario:")
        print(f"{'Arm':<18} {'Selection Rate':>15} {'Win Rate':>15}")
        print("-" * 50)

        # Aggregate win rates
        arm_stats = defaultdict(lambda: {'selections': 0, 'wins': 0, 'count': 0})

        for r in by_scenario[scenario]:
            sel_rates = r.get('selection_rates', {})
            win_rates = r.get('win_rates', {})

            for arm in sel_rates:
                arm_stats[arm]['selections'] += sel_rates.get(arm, 0)
                arm_stats[arm]['wins'] += win_rates.get(arm, 0)
                arm_stats[arm]['count'] += 1

        for arm in sorted(arm_stats.keys()):
            stats = arm_stats[arm]
            avg_sel = stats['selections'] / max(1, stats['count'])
            avg_win = stats['wins'] / max(1, stats['count'])
            print(f"{arm:<18} {avg_sel:>15.2%} {avg_win:>15.2%}")


def print_reproducibility(results: List[Dict]):
    """Print reproducibility analysis for seed variations."""
    print(f"\n{'='*80}")
    print("REPRODUCIBILITY CHECK (MDB Standard, Different Seeds)")
    print(f"{'='*80}")

    # Filter to MDB standard with base pool
    seed_results = [r for r in results
                   if r.get('config', {}).get('policy') == 'mdb'
                   and r.get('config', {}).get('scenario') == 'standard'
                   and set(r.get('config', {}).get('arms', [])) == {'random', 'single_feature', 'xgboost', 'linucb'}]

    if not seed_results:
        print("No reproducibility results found")
        return

    print(f"\n{'Seed':>8} {'Total Regret':>15} {'Final Set E':<25} {'Inferences':>12}")
    print("-" * 65)

    regrets = []
    for r in sorted(seed_results, key=lambda x: x['config'].get('seed', 0)):
        seed = r['config'].get('seed', 'N/A')
        regret = r['metrics']['total_ndcg_regret']
        set_e = r.get('policy_stats', {}).get('set_E', [])
        cost = r['inference_stats']['total_inferences']

        regrets.append(regret)
        print(f"{seed:>8} {regret:>15.1f} {str(set_e):<25} {cost:>12}")

    if len(regrets) > 1:
        import statistics
        print(f"\n{'Mean':>8} {statistics.mean(regrets):>15.1f}")
        print(f"{'Std Dev':>8} {statistics.stdev(regrets):>15.1f}")


def print_key_findings(results: List[Dict]):
    """Print key findings and insights."""
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")

    findings = []

    # 1. MDB vs Uniform regret comparison
    for scenario in ['standard', 'cascade', 'noisy']:
        mdb_results = [r for r in results
                      if r.get('config', {}).get('policy') == 'mdb'
                      and r.get('config', {}).get('scenario') == scenario
                      and set(r.get('config', {}).get('arms', [])) == {'random', 'single_feature', 'xgboost', 'linucb'}]

        uniform_results = [r for r in results
                         if r.get('config', {}).get('policy') == 'uniform'
                         and r.get('config', {}).get('scenario') == scenario
                         and set(r.get('config', {}).get('arms', [])) == {'random', 'single_feature', 'xgboost', 'linucb'}]

        if mdb_results and uniform_results:
            mdb_regret = mdb_results[0]['metrics']['total_ndcg_regret']
            uniform_regret = uniform_results[0]['metrics']['total_ndcg_regret']
            mdb_cost = mdb_results[0]['inference_stats']['total_inferences']
            uniform_cost = uniform_results[0]['inference_stats']['total_inferences']

            regret_reduction = (uniform_regret - mdb_regret) / uniform_regret * 100
            cost_reduction = (uniform_cost - mdb_cost) / uniform_cost * 100

            findings.append(f"[{scenario.upper()}] MDB reduces regret by {regret_reduction:.1f}% vs Uniform ({mdb_regret:.1f} vs {uniform_regret:.1f})")
            findings.append(f"[{scenario.upper()}] MDB reduces inference cost by {cost_reduction:.1f}% ({mdb_cost:.0f} vs {uniform_cost:.0f})")

    # 2. Best arm identification
    mdb_base = [r for r in results
               if r.get('config', {}).get('policy') == 'mdb'
               and set(r.get('config', {}).get('arms', [])) == {'random', 'single_feature', 'xgboost', 'linucb'}]

    winners = defaultdict(int)
    for r in mdb_base:
        set_e = r.get('policy_stats', {}).get('set_E', [])
        for arm in set_e:
            winners[arm] += 1

    if winners:
        best_arm = max(winners.items(), key=lambda x: x[1])
        findings.append(f"MDB consistently identifies '{best_arm[0]}' as the winner across {best_arm[1]} experiments")

    # 3. Learning arms performance
    learning_only = [r for r in results
                    if set(r.get('config', {}).get('arms', [])) == {'linucb', 'linear_ts'}
                    and r.get('config', {}).get('policy') == 'mdb']

    if learning_only:
        avg_regret = sum(r['metrics']['total_ndcg_regret'] for r in learning_only) / len(learning_only)
        findings.append(f"Learning arms only (linucb vs linear_ts): Average regret = {avg_regret:.1f}")

    print()
    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding}")


def main():
    """Main analysis function."""
    print("\n" + "="*80)
    print("MULTI-DUELING BANDIT EXPERIMENT ANALYSIS")
    print("="*80)

    results = load_results()

    if not results:
        print("No results found. Run experiments first.")
        return

    print(f"\nLoaded {len(results)} experiment results")

    # Categorize and analyze
    categories = categorize_results(results)

    # Policy comparisons by scenario (base pool)
    for scenario in ['standard', 'cascade', 'noisy']:
        key = f"{scenario}_base_pool"
        if key in categories:
            print_policy_comparison(categories[key], scenario, "Base Pool: random, single_feature, xgboost, linucb")

    # Full arm pool comparison
    for scenario in ['standard', 'cascade', 'noisy']:
        key = f"{scenario}_full_pool"
        if key in categories:
            print_policy_comparison(categories[key], scenario, "Full Pool: +linear_ts")

    # Learning arms only
    for scenario in ['standard', 'cascade', 'noisy']:
        key = f"{scenario}_learning_only"
        if key in categories:
            print_policy_comparison(categories[key], scenario, "Learning Only: linucb, linear_ts")

    # MDB convergence
    print_mdb_convergence(results)

    # Arm performance
    print_arm_performance(results)

    # Reproducibility
    print_reproducibility(results)

    # Key findings
    print_key_findings(results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
