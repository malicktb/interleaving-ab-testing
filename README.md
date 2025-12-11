# H-MDB-KT: Hierarchical Multi-Dueling Bandits with Knowledge Transfer

Accelerate online ranker evaluation by replacing slow A/B testing with Hierarchical Multi-Dueling Bandits (H-MDB-KT). This framework reduces inference costs by 46-86% while achieving lower cumulative regret than flat MDB approaches.

## Glossary

| Term | Full Form | Meaning |
|------|-----------|---------|
| **MDB** | Multi-Dueling Bandit | Algorithm for comparing K ranking arms simultaneously using pairwise win statistics |
| **H-MDB** | Hierarchical MDB | Two-level MDB that evaluates cluster representatives first, then winning cluster members |
| **H-MDB-KT** | H-MDB with Knowledge Transfer | H-MDB enhanced with warm-start initialization to prevent cold-start regret spikes |
| **KT** | Knowledge Transfer | Mechanism that initializes cluster members using their representative's statistics |
| **UCB** | Upper Confidence Bound | Exploration strategy that balances exploitation with exploration of uncertain arms |
| **LCB** | Lower Confidence Bound | Conservative bound used to identify proven winners (Set E) |
| **DCG** | Discounted Cumulative Gain | Ranking quality metric: `rel / log2(pos + 1)` |
| **NDCG** | Normalized DCG | DCG / Ideal DCG, producing a score in [0, 1] |
| **TS** | Thompson Sampling | Bayesian exploration via posterior sampling |
| **PBM** | Position-Based Model | Click model where examination probability depends on position |
| **PCA** | Principal Component Analysis | Dimensionality reduction (700D → 20D features) |
| **LTR** | Learning to Rank | ML task of ordering documents by relevance |
| **Slate** | Result Slate | Final ordered list of documents shown to user, created via Team Draft interleaving |

## The Problem

Evaluating **K candidate rankers** with traditional methods has critical drawbacks:

**A/B Testing Limitations:**
- Sequential testing (K tests vs control) takes weeks
- Traffic splitting reduces statistical power per variant
- Users in losing buckets experience inferior results (cumulative regret)
- No early stopping for underperforming variants

**Flat MDB Limitations:**
- Requires O(K) model inferences per query
- Prohibitive latency when K=100+ candidates
- The "Efficiency Trilemma": cannot simultaneously achieve Scale, Latency, and Quality

## The Solution: H-MDB-KT

H-MDB-KT addresses these issues through hierarchical evaluation with knowledge transfer:

```
H-MDB-KT Architecture                           
1. OFFLINE PHASE: Cluster arms by output similarity (Jaccard)
2. LEVEL 1: Evaluate M cluster representatives (M << K)
3. KNOWLEDGE TRANSFER: Copy winner's W/N stats to members 
4. LEVEL 2: Evaluate winning cluster members with warm start 
```

### Key Benefits

| Metric | Flat MDB | H-MDB-KT | Improvement |
|--------|----------|----------|-------------|
| Arms/Query | 2.86 | 1.54 | 46% reduction |
| Cumulative Regret | 2410.4 | 2219.8 | 7.9% lower |
| Inference Cost (K=100) | 1,000,000 | 144,898 | 85.5% reduction |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test run
python run_experiment.py --quick

# Standard flat MDB experiment
python run_experiment.py --policy mdb --rounds 5000

# H-MDB-KT hierarchical experiment
python run_experiment.py --hierarchical --policy mdb --arm-pool full --rounds 10000

# Scalability experiment (K=100 arms)
python run_experiment.py --hierarchical --arm-pool full --arm-subset 100 --rounds 10000
```

## CLI Reference

### Basic Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--policy` | `mdb` | Policy algorithm: `mdb`, `multi_rucb`, `uniform`, `ts`/`thompson`, `fixed:ARM_NAME` |
| `--scenario` | `standard` | Click model: `standard` (PBM), `cascade`, `noisy` |
| `--rounds` | `5000` | Number of simulation rounds (queries) |
| `--arms` | `[random, single_feature, xgboost, linucb]` | Arm types to include |
| `--seed` | `42` | Random seed for reproducibility |
| `--quick` | - | Quick test mode (1000 rounds, 1000 train records) |

### Arm Pool Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--arm-pool` | `small` | Pool size: `small` (uses --arms), `medium` (~42), `full` (~100) |
| `--arm-subset` | `None` | Select K arms from full pool (e.g., `--arm-subset 20` for K=20) |

### Hierarchical (H-MDB-KT) Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--hierarchical` | `False` | Enable two-level hierarchical evaluation |
| `--level1-rounds` | `2000` | Rounds before Level 1→2 transition |
| `--clustering` | `output_based` | Method: `output_based` (Jaccard) or `random` (ablation) |
| `--warm-start` | `True` | Enable Knowledge Transfer at level transition |
| `--no-warm-start` | - | Disable KT (cold start ablation) |
| `--cluster-min-size` | `5` | Minimum cluster size for HDBSCAN |
| `--cluster-sample-queries` | `1000` | Queries for Jaccard similarity computation |
| `--cluster-top-k` | `10` | Top-k documents for similarity |

### Statistics Tracker Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tracker` | `cumulative` | Tracker type: `cumulative` or `discounted` |
| `--discount-factor` | `0.995` | Decay factor γ for discounted tracker |

### Attribution & Multileaving Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--multileaving` | `team_draft` | Multileaving scheme |
| `--attribution` | `team_draft_legacy` | Attribution scheme for credit assignment |

### MultiRUCB Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--multi-rucb-m` | `None` | Comparison set size (default: all arms) |

### Score Cache Arguments (K=100 Scalability)

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-cache` | `False` | Disable score pre-computation cache |
| `--cache-threshold` | `10` | Only use cache if ≥ this many arms |

### Model Persistence Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--save-arms` | `None` | Save trained arms to directory for reuse |
| `--load-arms` | `None` | Load pre-trained arms (skips training) |
| `--logfile` | `None` | Specific output file path (overrides timestamp naming) |

## Example Experiments

### 1. Basic Policy Comparison

```bash
# Flat MDB (baseline)
python run_experiment.py --policy mdb --scenario standard --rounds 5000

# Uniform exploration (no elimination)
python run_experiment.py --policy uniform --scenario standard --rounds 5000

# Thompson Sampling (traditional A/B)
python run_experiment.py --policy ts --scenario standard --rounds 5000

# Fixed arm (bound calculation)
python run_experiment.py --policy fixed:xgboost --scenario standard --rounds 5000
```

### 2. H-MDB-KT Experiments (RQ1: Inference Efficiency)

```bash
# H-MDB-KT with output-based clustering
python run_experiment.py \
    --hierarchical \
    --policy mdb \
    --arm-pool full \
    --level1-rounds 2000 \
    --rounds 10000

# Compare with Flat MDB baseline
python run_experiment.py \
    --policy mdb \
    --arm-pool full \
    --rounds 10000
```

### 3. Knowledge Transfer Ablation (RQ2)

```bash
# With Knowledge Transfer (warm start)
python run_experiment.py \
    --hierarchical \
    --warm-start \
    --arm-pool full \
    --rounds 10000

# Without Knowledge Transfer (cold start)
python run_experiment.py \
    --hierarchical \
    --no-warm-start \
    --arm-pool full \
    --rounds 10000
```

### 4. Clustering Validity (RQ3)

```bash
# Output-based clustering (H-MDB-KT)
python run_experiment.py \
    --hierarchical \
    --clustering output_based \
    --arm-pool full \
    --rounds 10000

# Random clustering (Random-KT ablation)
python run_experiment.py \
    --hierarchical \
    --clustering random \
    --arm-pool full \
    --rounds 10000
```

### 5. Scalability Sweep (K=20, 60, 100)

```bash
# K=20 arms
python run_experiment.py \
    --hierarchical \
    --arm-pool full \
    --arm-subset 20 \
    --rounds 10000

# K=60 arms
python run_experiment.py \
    --hierarchical \
    --arm-pool full \
    --arm-subset 60 \
    --rounds 10000

# K=100 arms (full scale)
python run_experiment.py \
    --hierarchical \
    --arm-pool full \
    --arm-subset 100 \
    --rounds 10000
```

### 6. Click Model Robustness (Environmental Scenarios)

```bash
# Standard (Position-Based Model)
python run_experiment.py --hierarchical --scenario standard --rounds 10000

# Cascade (sparse feedback)
python run_experiment.py --hierarchical --scenario cascade --rounds 10000

# Noisy (robustness test)
python run_experiment.py --hierarchical --scenario noisy --rounds 10000
```

### 7. Non-Stationary Arms with Discounted Tracker

```bash
# Discounted statistics for learning arms
python run_experiment.py \
    --policy mdb \
    --tracker discounted \
    --discount-factor 0.995 \
    --arms linucb linear_ts xgboost \
    --rounds 10000
```

### 8. Reproducibility with Multiple Seeds

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
    python run_experiment.py \
        --hierarchical \
        --seed $seed \
        --logfile "results/hmdb_seed_${seed}.json"
done
```

### 9. Pre-training Arms for Repeated Experiments

```bash
# Train and save arms once
python run_experiment.py \
    --arm-pool full \
    --save-arms saved_models/full_pool \
    --rounds 1000

# Reuse trained arms for multiple experiments
python run_experiment.py \
    --hierarchical \
    --load-arms saved_models/full_pool \
    --rounds 10000
```

## Architecture

```
run_experiment.py              # Main entry point

config.py                      # ExperimentConfig dataclass

core/                          # Shared base classes & utilities
  base/
    arm.py                     # BaseArm ABC
    policy.py                  # BasePolicy ABC
    click_model.py             # ClickSimulator ABC
    attribution.py             # BaseAttributionStrategy ABC
    statistics.py              # StatisticsTrackerBase ABC
  types.py                     # QueryRecord, AttributionResult, ClusteringResult
  metrics.py                   # DCG/NDCG, RegretTelemetry
  profiler.py                  # Inference tracking

src/                           # Algorithm implementations
  arms/                        # Ranking algorithms
    factory.py                 # create_arm_pool()
    random.py, xgboost.py, linucb.py, linear_ts.py, ...
    ground_truth.py

  policies/                    # Bandit policies
    mdb.py                     # Multi-Dueling Bandit
    multi_rucb.py              # MultiRUCB alternative
    hierarchical_mdb.py        # H-MDB-KT (two-level)
    baseline.py                # Uniform, Thompson, Fixed
    trackers/                  # W/N matrix tracking
      cumulative.py
      discounted.py

  multileaving/                # Slate construction
    team_draft.py              # Team Draft interleaving
    factory.py                 # Multileaving factory
    attribution/               # Credit assignment

  clustering/                  # H-MDB clustering
    output_based.py            # Jaccard similarity + HDBSCAN

simulation/                    # Experiment execution
  simulator.py                 # Core simulation engine
  data/
    loader.py                  # Yahoo LTR data loading
    cache.py                   # Score pre-computation (K=100)
  click_models/
    pbm.py, cascade.py, noisy.py

analysis/                      # Results analysis
  scripts/
    analyze_results.py
    statistical_tests.py
```

## H-MDB-KT Algorithm Details

### Output-Based Clustering

Arms are clustered by behavioral similarity using Jaccard index on top-k outputs:

```
J(i, j) = (1/|Q|) Σ |top_k(Arm_i, q) ∩ top_k(Arm_j, q)|
                   |top_k(Arm_i, q) ∪ top_k(Arm_j, q)|
```

Where Q is a reference query set (default: 1000 queries) and k=10.

Clustering uses HDBSCAN with `min_cluster_size=5`. Representatives are selected by highest offline NDCG@5 within each cluster.

### Two-Level Hierarchy

**Level 1 (Exploration):** Evaluate only M cluster representatives
- Active set: A_t ⊆ {ρ_1, ..., ρ_M} where M << K
- Duration: First `level1_rounds` (default: 2000)
- Goal: Identify winning cluster

**Level 2 (Exploitation):** Evaluate winning cluster members
- Active set: A_t ⊆ C* (winning cluster)
- Duration: Remaining rounds
- Benefits from Knowledge Transfer

### Knowledge Transfer (Warm Start)

At level transition, member statistics are initialized from representative:

```
W_{a,j} ← W_{ρ_win,j}    N_{a,j} ← N_{ρ_win,j}
W_{j,a} ← W_{j,ρ_win}    N_{j,a} ← N_{j,ρ_win}
```

This prevents the cold-start regret spike by providing informed priors.

### UCB Bounds

```
UCB[i,j] = W[i,j]/N[i,j] + √(α·log(t)/N[i,j])
LCB[i,j] = W[i,j]/N[i,j] - √(α·log(t)/N[i,j])

Set E (Proven Winners): {i : LCB[i,j] > 0.5, ∀j ≠ i}
Set F (Potential Winners): {i : UCB[i,j] > 0.5, ∃j ≠ i}
```

### Grace Period

Learning arms (LinUCB, LinearTS) need protection during initial convergence:

```
Eliminate i ⟺ (∃j : UCB[i,j] < 0.5) ∧ (N[i,j] ≥ n_min)
```

Default `n_min=500` rounds before elimination can occur.

## Arms (Ranking Algorithms)

### Static Arms (K=100 Pool)

| Type | Count | Description |
|------|-------|-------------|
| XGBoost | 27 | Grid: depth∈{3,6,9}, lr∈{0.01,0.1,0.3}, n_est∈{50,100,200} |
| Single Feature | 10 | Rank by single raw feature |
| Random | 1 | Uniform random permutation |

### Learning Arms

| Arm | Description |
|-----|-------------|
| LinUCB (4) | Contextual bandit with α∈{0.1, 0.5, 1.0, 2.0} |
| LinearTS (1) | Thompson Sampling with Bayesian linear regression |

## Click Models

| Model | Behavior | Use Case |
|-------|----------|----------|
| **PBM** | Independent clicks, position decay | Standard evaluation |
| **Cascade** | Click first relevant, stop | Sparse feedback testing |
| **Noisy** | 10% random + false negatives | Robustness testing |

## Target Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| Cumulative Regret | Σ(NDCG* - NDCG_shown) | Lower is better |
| Arms/Query | Average |A_t| per round | Lower is better |
| Cost Reduction | 1 - (Cost_H-MDB / Cost_Flat) | Higher is better |
| Total Inferences | Σ|A_t| over T rounds | Lower is better |

## Experimental Parameters (Defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Horizon T | 10,000 | Total rounds |
| Slate size | 10 | Documents per page |
| UCB α | 0.51 | Exploration parameter |
| Grace period | 500 | Rounds before elimination |
| Level 1 duration | 2,000 | Rounds before Level 2 |
| Arm pool K | 100 | Full pool size |
| Reference set |Q| | 1,000 | Queries for clustering |

## Output Format

Results are saved to `evaluation_results/` as JSON:

```python
{
    "config": {
        "scenario": "standard",
        "policy": "mdb",
        "hierarchical": True,
        "level1_rounds": 2000,
        ...
    },
    "metrics": {
        "iterations": 10000,
        "total_ndcg_regret": 2219.8,
        "avg_ndcg_regret": 0.222,
        "click_rate": 0.68
    },
    "logical_cost": {
        "total_evaluations": 15400,
        "avg_arms_per_query": 1.54
    },
    "policy_stats": {
        "set_E": ["xgboost_d6_lr0.1_n100"],
        "set_F": [...]
    },
    "runtime": 127.3
}
```

## References

| Algorithm | Paper |
|-----------|-------|
| **Multi-Dueling Bandits** | Brost et al., "Multi-dueling bandits with dependent arms" (SIGIR 2016) |
| **MultiRUCB** | Du et al., "Multi-dueling bandits and their application to online ranker evaluation" (CIKM 2021) |
| **LinUCB** | Li et al., "A contextual-bandit approach to personalized news article recommendation" (WWW 2010) |
| **Multileaving (Team Draft)** | Schuth et al., "Multileaved comparisons for fast online evaluation" (CIKM 2014) |
| **HDBSCAN** | Campello et al., "Density-based clustering based on hierarchical density estimates" (PAKDD 2013) |
| **Yahoo LTR Dataset** | Chapelle & Chang, "Yahoo! learning to rank challenge overview" (JMLR 2011) |
