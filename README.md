# Multi-Dueling Bandit Framework

Accelerate online ranker evaluation by replacing slow A/B testing with Multi-Dueling Bandits (MDB).

## Glossary

| Term | Full Form | Meaning |
|------|-----------|---------|
| **MDB** | Multi-Dueling Bandit | Algorithm for comparing K ranking arms simultaneously using pairwise win statistics |
| **UCB** | Upper Confidence Bound | Exploration strategy that balances exploitation of known good arms with exploration of uncertain ones |
| **DCG** | Discounted Cumulative Gain | Ranking quality metric that weights relevance scores by position: `rel / log2(pos + 1)` |
| **NDCG** | Normalized DCG | DCG divided by ideal DCG, producing a score in [0, 1] where 1.0 = perfect ranking |
| **TS** | Thompson Sampling | Bayesian exploration strategy that samples from posterior distributions to select arms |
| **PBM** | Position-Based Model | Click simulation model where examination probability depends on result position |
| **PCA** | Principal Component Analysis | Dimensionality reduction technique (used here: 700D → 20D features) |
| **LTR** | Learning to Rank | Machine learning task of ordering documents by relevance to a query |
| **Slate** | Result Slate | The final ordered list of documents shown to the user. In multileaving, the slate is created by interleaving rankings from multiple arms via Team Draft, blending contributions from all competing rankers into one unified result page. |

## The Problem

Evaluating **K candidate rankers** with traditional A/B testing has drawbacks:
- **Sequential testing** (K tests vs control) is slow
- **Multi-arm A/B/n** splits traffic, reducing statistical power per variant
- **Users in losing buckets** experience inferior results (cumulative regret)
- **No early stopping** - underperforming variants run for the full test duration

## The Solution

MDB addresses these issues through adaptive multileaving:

```
1. MULTILEAVE  →  Combine rankings from multiple arms into ONE slate
2. ATTRIBUTE   →  User clicks → credit goes to the responsible arm
3. ELIMINATE   →  UCB (Upper Confidence Bound) bounds prune losing arms early
```

## Core Objectives

1. **Sample Efficiency** — Identify the best ranker using fewer user impressions than A/B testing

2. **Regret Minimization** — Reduce cumulative quality loss (NDCG, Normalized Discounted Cumulative Gain, gap vs Ground Truth) by eliminating bad arms early

3. **Handle Non-Stationary Arms** — Prove MDB works even when some arms (LinUCB) are still learning during evaluation (via grace period protection)

## Components

| Component | Purpose |
|-----------|---------|
| **Arms** (`src/arms/`) | 6 ranking algorithms compete |
| **MDB Policy** (`src/policies/mdb.py`) | Decides which arms to test each round using UCB confidence bounds |
| **Team Draft** (`src/multileaving/`) | Fairly interleaves rankings into one result page |
| **Click Models** (`src/environment/click_models/`) | Simulates user behavior (PBM/Position-Based Model, Cascade, Noisy) |
| **Simulator** (`src/environment/simulator.py`) | Runs the full experiment loop on Yahoo LTR (Learning to Rank) data |

## Arms (Ranking Algorithms)

In multi-arm bandit terminology, each arm is a candidate ranking algorithm competing to be identified as the best. When an arm is "pulled" (selected), it produces a ranking for the current query, and receives reward signal from user clicks.

| Arm | Type | File | Description |
|-----|------|------|-------------|
| `random` | Baseline | `random.py` | **Random Ranker** — Shuffles documents randomly. Lower bound baseline with no learning. Used to measure how much better other arms perform. |
| `single_feature` | Baseline | `single_feature.py` | **Single Feature Ranker** — Ranks by one pre-computed feature (e.g., BM25 score at feature index 0). Static heuristic baseline. |
| `ground_truth` | Ground Truth | `ground_truth.py` | **Ground Truth Ranker** — Ranks documents by Yahoo LTR human-annotated relevance labels (0-4 scale). Represents the ideal ranking. Not a competing arm; used to compute regret (quality gap between shown slate and perfect ranking). |
| `xgboost` | Supervised | `xgboost.py` | **XGBoost Ranker** — Gradient boosted trees trained offline on historical data. Strong static baseline that doesn't adapt online. |
| `linucb` | Contextual Bandit | `linucb.py` | **LinUCB Ranker** — Online contextual bandit using Upper Confidence Bound. Learns from click feedback during evaluation. Uses PCA (700D → 20D) for efficiency. |
| `linear_ts` | Contextual Bandit | `linear_ts.py` | **Linear Thompson Sampling (TS)** — Bayesian alternative to LinUCB. Samples from posterior distribution for exploration. Also learns online from clicks. |

### Arm Categories

**Static Arms** (don't learn during evaluation):
- `random`, `single_feature`, `xgboost` — Fixed ranking policy throughout experiment

**Learning Arms** (improve during evaluation):
- `linucb`, `linear_ts` — Update their models from click feedback, requiring grace period protection to avoid premature elimination

**Ground Truth** (for measurement only):
- `ground_truth` — Ranks by Yahoo dataset's human-annotated relevance labels (0=irrelevant to 4=highly relevant). Provides ideal ranking for NDCG regret calculation. The relevance grades were assigned by human editors judging query-document pairs.

## Multileaving (`src/multileaving/`)

Multileaving is the technique that enables comparing multiple rankers simultaneously using a single result page. Instead of showing users results from just one ranker (as in A/B testing), multileaving combines rankings from all competing arms into one interleaved **slate**.

### What is a Slate?

A **slate** is the final ordered list of documents presented to the user for a single query. In traditional A/B testing, each user sees a slate from exactly one ranker. In multileaving, the slate is a blend:

```
Traditional A/B:
  User 1 sees: [Ranker A's results]
  User 2 sees: [Ranker B's results]

Multileaving:
  All users see: [Blended slate from A + B + C]
                  ↑
                  Documents interleaved via Team Draft
                  Each position tracked back to its source arm
```

The slate maintains an **attribution map** recording which arm contributed each document, enabling credit assignment when users click.

### How It Works

```
Query arrives
    ↓
Each active arm produces a ranking: [doc1, doc2, doc3, ...]
    ↓
Team Draft combines them into ONE slate
    ↓
User sees blended results and clicks
    ↓
Attribution maps click → responsible arm
    ↓
Policy updates pairwise win statistics
```

### Components

| File | Function | Description |
|------|----------|-------------|
| `team_draft.py` | `interleave()` | **Team Draft Algorithm** — Fairly combines multiple rankings into one slate. Arms take turns picking their top-ranked items in round-robin fashion, with order reversal each round to prevent position bias. |
| `attribution.py` | `get_click_winner()` | **Click Attribution** — Maps the first clicked item back to the arm that contributed it. Returns the "winner" of this query. |
| `attribution.py` | `compute_credit()` | **Credit Assignment** — Supports two models: *navigational* (first click wins) and *informational* (all clicks count). |

### Team Draft Algorithm

```
Input: Rankings from arms A, B, C
       A: [5, 2, 8, 1]
       B: [2, 5, 3, 7]
       C: [8, 2, 5, 4]

Round 1 (order: A→B→C):
  A picks 5, B picks 2, C picks 8

Round 2 (order: C→B→A, reversed):
  C picks 4, B picks 3, A picks 1

Final Slate: [5, 2, 8, 4, 3, 1]
Attribution: {5:A, 2:B, 8:C, 4:C, 3:B, 1:A}
```

### Why Multileaving?

| Traditional A/B | Multileaving |
|-----------------|--------------|
| One ranker per user | All rankers contribute to same slate |
| Need separate traffic buckets | Single unified experience |
| Slow pairwise comparisons | Simultaneous K-way comparison |
| Users stuck with bad rankers | Blended quality from all rankers |

## Policies (`src/policies/`)

A policy is the decision-maker that controls which arms compete in each round. It maintains a scoreboard of pairwise win/loss statistics and uses this data to decide whether to explore (test more arms) or exploit (use the best arm).

### How Policies Fit In

```
Each Round:
    ↓
Policy.select_arms() → Which arms compete?
    ↓
Arms produce rankings → Multileaved into slate
    ↓
User clicks → Winner determined
    ↓
Policy.update(winner, participants) → Update W/N matrices
```

### Components

| File | Class | Description |
|------|-------|-------------|
| `base.py` | `BasePolicy` | **Abstract base class** — Maintains pairwise Win matrix (W) and Comparison matrix (N). Implements `update()` logic for recording wins. |
| `mdb.py` | `MDBPolicy` | **Multi-Dueling Bandit** — The core algorithm. Uses UCB confidence bounds to maintain Set E (proven winners) and Set F (potential contenders). Eliminates losing arms early. |
| `baseline.py` | `UniformPolicy` | **Pure exploration** — Always selects ALL arms. No pruning. Baseline for measuring MDB's speedup. |
| `baseline.py` | `FixedPolicy` | **No exploration** — Always selects ONE specific arm. Used for ground truth/random bounds. |
| `baseline.py` | `SingleArmThompsonSamplingPolicy` | **Traditional A/B** — Selects ONE arm per round via Thompson Sampling. Baseline for comparing MDB vs standard bandits. |

### MDB Algorithm (Brost et al., 2016)

MDB maintains two sets using Upper Confidence Bounds:

```
Set E (Proven Winners):
  Arms that beat ALL others with ≥50% confidence (narrow bound)

Set F (Potential Contenders):
  Arms that MIGHT be best (wide bound, more inclusive)

Decision Rule:
  if |E| == 1:  → Exploit: use only the winner
  else:         → Explore: test all arms in F
```

**UCB Formula:**
```
UCB[i,j] = W[i,j]/N[i,j] + √(α·log(t)/N[i,j])

Where:
  W[i,j] = times arm i beat arm j
  N[i,j] = times arm i faced arm j
  α = exploration parameter (default 0.51)
  t = total rounds
```

### Grace Period

Learning arms (LinUCB, LinearTS) start with poor performance and improve over time. Without protection, MDB would eliminate them before they learn.

**Solution:** During the first `n_min` rounds (default 500), all arms are forced into Set F. No elimination occurs.

```python
MDBPolicy(arm_names, alpha=0.51, beta=1.0, n_min=500)
```

### Policy Comparison

| Policy | Arms Selected | Use Case |
|--------|---------------|----------|
| `MDBPolicy` | Adaptive (1 to K) | Main algorithm — fast convergence |
| `UniformPolicy` | Always K | Baseline — no intelligence |
| `FixedPolicy` | Always 1 (fixed) | Bounds — ground truth/random performance |
| `SingleArmThompsonSamplingPolicy` | Always 1 (sampled) | Baseline — traditional A/B |

## Click Models (`src/environment/click_models/`)

**Click Models** simulate user behavior in response to search result slates. Since we can't run live experiments, these models generate realistic click feedback based on item relevance and position.

### Why Click Models Matter

In a real system, users click on results based on:
1. **Position** — Users examine top results more often (position bias)
2. **Relevance** — Users click relevant items more than irrelevant ones
3. **Behavior patterns** — Some users scan exhaustively, others stop after first good result

Click models capture these patterns to generate training signal for the MDB algorithm.

### Components

| File | Class | Description |
|------|-------|-------------|
| `base.py` | `ClickSimulator` | **Abstract base class** — Defines `simulate()` interface returning list of clicked positions. |
| `pbm.py` | `PositionBasedModel` | **Position-Based Model** — Independent examination per position. Multiple clicks possible. |
| `cascade.py` | `CascadeModel` | **Cascade Model** — User scans top-down, clicks first relevant item, stops. Single click max. |
| `noisy.py` | `NoisyUserModel` | **Noisy User** — Cascade with random noise and false negatives. Tests robustness. |

### Position-Based Model (PBM)

Most realistic model. Each position is examined independently with decaying probability.

```
P(click at position i) = P(examine | i) × P(click | relevance)

Examination decay:  [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
Click probabilities: {0: 0.01, 1: 0.10, 2: 0.30, 3: 0.60, 4: 0.95}
```

**Behavior:** User may click multiple items. Top positions get more attention.

### Cascade Model

Deterministic navigational behavior. User wants ONE answer.

```
For each position (top to bottom):
    if relevance >= threshold (default 3):
        CLICK and STOP
    else:
        continue scanning
```

**Behavior:** At most one click. Sparse feedback. Tests MDB with limited signal.

### Noisy User Model

Realistic noise for robustness testing.

```
With probability noise_prob (10%):
    Click random position (irrational)
Otherwise:
    Cascade behavior with false_negative_rate (10%)
    (may skip a relevant item)
```

**Behavior:** Introduces attribution errors. Tests whether MDB converges despite noise.

### How Click Models Connect to Simulation

```
Simulator Loop:
    ↓
Team Draft produces slate: [doc5, doc2, doc8, ...]
    ↓
Click Model receives: slate + relevance grades
    ↓
PositionBasedModel.simulate() → [0, 2]  (clicked positions)
    ↓
Attribution: position 0 → arm that contributed doc5
    ↓
Policy.update(winner="xgboost", participants=[...])
```

### Experiment Scenarios

The `config/scenarios.py` file defines three standard scenarios:

| Scenario | Click Model | Use Case |
|----------|-------------|----------|
| `standard` | PositionBasedModel | Default. Realistic multi-click behavior. |
| `cascade` | CascadeModel | Sparse feedback. Tests sample efficiency. |
| `noisy` | NoisyUserModel | Robustness. Tests convergence with errors. |

## Simulator (`src/environment/simulator.py`)

Simulator is the orchestration layer that ties all components together. It implements a Gym-like interface for running MDB experiments, coordinating the flow between policies, arms, multileaving, and click models.

### Why a Simulator?

The MDB algorithm requires coordination between multiple systems:
- **Policy** decides which arms compete
- **Arms** generate rankings for queries
- **Multileaving** combines rankings into one slate
- **Click Model** generates user feedback
- **Attribution** maps clicks back to arms
- **Telemetry** records metrics for analysis

The Simulator handles this orchestration and provides a clean `step()` API.

### Core Methods

| Method | Description |
|--------|-------------|
| `__init__()` | Accepts arms, policy, data_loader, click_model, ground_truth, slate_size |
| `reset()` | Initializes RegretTelemetry and Profiler for new episode |
| `train_arms()` | Offline training phase — trains all arms on historical data |
| `step(record)` | Process one query — the core simulation loop |
| `run_episode()` | Iterate over test data calling `step()` for each query |
| `get_results()` | Return comprehensive metrics, win rates, and history |

### The `step()` Function — Core Loop

Each call to `step(record)` executes one iteration of the MDB algorithm:

```
step(QueryRecord):
    │
    ├─1─► Policy.select_arms()
    │     → ["xgboost", "linucb", "linear_ts"]
    │
    ├─2─► For each active arm:
    │         arm.rank(record) → [doc3, doc1, doc7, ...]
    │         Profiler.increment_inference()
    │
    ├─3─► interleave(rankings)
    │     → slate: [doc3, doc5, doc1, ...]
    │     → attribution: {0: "xgboost", 1: "linucb", ...}
    │
    ├─4─► ClickModel.simulate(slate, relevance)
    │     → clicks: [0, 2]  (positions clicked)
    │
    ├─5─► get_click_winner(clicks, slate, attribution)
    │     → winner: "xgboost"
    │
    ├─6─► Policy.update(winner, participants)
    │     → Updates W/N matrices
    │
    ├─7─► Learning arms update on clicked features
    │     → LinUCB/LinearTS learn from positive example
    │
    └─8─► RegretTelemetry.record_ndcg_regret()
          → Tracks Ground Truth NDCG - Shown NDCG
```

### Two-Phase Experiment

```
Phase 1: Offline Training
    └─► train_arms(train_records)
        └─► XGBoost trains on labeled data
        └─► LinUCB/LinearTS initialize parameters

Phase 2: Online Evaluation
    └─► run_episode(test_records)
        └─► For each query: step()
        └─► MDB eliminates losing arms over time
        └─► Learning arms improve from click feedback
```

### Results Structure

`get_results()` returns:

```python
{
    "metrics": {
        "iterations": 5000,
        "total_ndcg_regret": 127.3,
        "avg_ndcg_regret": 0.025,
        "click_rate": 0.68,
    },
    "selection_rates": {"xgboost": 0.95, "linucb": 0.45, ...},
    "win_rates": {"xgboost": 0.52, "linucb": 0.31, ...},
    "policy_stats": {"set_E": ["xgboost"], "set_F": ["xgboost"]},
    "inference_stats": {"total_inferences": 12500, "avg_arms_per_query": 2.5},
    "history": {"cumulative_ndcg_regret": [0.02, 0.05, 0.08, ...]},
}
```

### Key Design Decisions

1. **Shared Context Learning** — All learning arms update on clicked features, not just the winning arm. Standard practice for simulation environments.

2. **Ground Truth for Regret** — The ground truth arm uses Yahoo dataset's human-annotated relevance labels to compute ideal NDCG. Not a competing arm; purely for measuring quality gap.

3. **Profiler Integration** — Every `rank()` call increments the inference counter, proving MDB's compute savings.

4. **Navigational Attribution** — First click determines winner (single-click assumption for simplicity).

## Utils (`src/utils/`)

Utils provides metrics computation and profiling infrastructure for measuring experiment quality and computational cost.

### What is Regret?

In bandit algorithms, **regret** measures the cumulative cost of not always choosing the best option. For ranking:

```
Regret per query = (Quality of ideal ranking) - (Quality of shown ranking)
                 = Ground Truth NDCG - Shown Slate NDCG
```

The **Ground Truth** ranking comes from the Yahoo LTR dataset, where human editors assigned relevance grades (0-4) to each query-document pair. Sorting by these labels gives the ideal ranking that any algorithm should aspire to match.

- **Zero regret** means we showed the perfect ranking every time
- **High regret** means we showed suboptimal rankings (lost user satisfaction)
- **Cumulative regret** sums this gap over all queries — lower is better

MDB minimizes regret by quickly eliminating bad arms, so users see better results sooner.

### Components

| File | Component | What It Does |
|------|-----------|--------------|
| `metrics.py` | `compute_dcg(relevance, k)` | Computes DCG@k for a ranking. Takes relevance scores in rank order, returns weighted sum where top positions count more. Formula: `Σ rel[i] / log2(i + 2)` |
| `metrics.py` | `compute_ndcg(relevance, ideal, k)` | Computes NDCG@k = DCG / Ideal DCG. Normalizes to [0, 1] so rankings can be compared across queries with different relevance distributions. |
| `metrics.py` | `RegretTelemetry` | Records per-iteration data: which arms competed, who won, NDCG regret. Maintains running totals and time-series history for analysis. |
| `profiler.py` | `Profiler` | Counter tracking how many ranking inferences occurred. Used to measure MDB's computational savings vs uniform exploration. |

### RegretTelemetry in Detail

Records everything needed to analyze experiment quality:

```python
telemetry = RegretTelemetry(arm_names=["xgboost", "linucb", "random"])

# Called each iteration by Simulator
telemetry.record_iteration(
    selected_arms=["xgboost", "linucb"],  # Which arms competed
    winner="xgboost"                       # Who won the click attribution
)
telemetry.record_ndcg_regret(
    optimal_ndcg=0.95,  # What perfect ranking would score
    shown_ndcg=0.82     # What the blended slate scored
)
# → Regret for this query: 0.95 - 0.82 = 0.13

# After experiment
summary = telemetry.get_summary()
# → {"total_ndcg_regret": 127.3, "avg_ndcg_regret": 0.025, ...}
```

**Key Outputs:**
- `cumulative_ndcg_regret_history` — Time-series for plotting regret growth over iterations
- `arm_selection_counts` — How often each arm was included in multileaving
- `arm_win_counts` — How often each arm won click attribution

### Profiler in Detail

Class that counts ranking operations to measure computational efficiency:

```python
# Inside Simulator.step():
for arm_name in active_arms:
    arm.rank(record)
    Profiler.increment_inference(1, arm_name=arm_name)

Profiler.increment_query()  # One query processed

# After experiment
stats = Profiler.get_statistics()
# → {"total_inferences": 12500, "avg_arms_per_query": 2.5, ...}
```

**Why This Matters:**

| Scenario | Inferences per Query | Total for 5000 Queries |
|----------|---------------------|------------------------|
| Uniform (all 6 arms) | 6 | 30,000 |
| MDB (converges at iter 1000) | 6→1 | ~7,000 |

MDB's `avg_arms_per_query` starts at K (all arms) and drops toward 1 as losers are eliminated.

### How Utils Connects to Simulation

```
Each step() call:
    │
    ├─► Profiler.increment_query()
    │
    ├─► For each arm selected:
    │       Profiler.increment_inference(arm_name)
    │
    ├─► Ground Truth computes ideal NDCG for this query
    │
    ├─► Shown slate's NDCG computed
    │
    └─► RegretTelemetry.record_ndcg_regret(optimal, shown)
        RegretTelemetry.record_iteration(arms, winner)
```

## Configuration (`config/`)

The `config/` directory contains settings that control experiment behavior without modifying code.

### ExperimentConfig (`config/experiments.py`)

Central configuration dataclass with all tunable parameters:

```python
@dataclass
class ExperimentConfig:
    # --- Experiment Setup ---
    n_rounds: int = 10000           # Total queries to process
    random_seed: int = 42           # For reproducibility
    arm_pool_list: list = ["random", "single_feature", "xgboost", "linucb"]

    # --- Environment ---
    data_path: str = "data/processed/yahoo_parquet/set1"
    scenario: str = "standard"      # Click model: "standard", "cascade", or "noisy"
    max_train_records: int = 20000  # Queries for offline training phase

    # --- MDB Policy Parameters ---
    strategy_alpha: float = 0.51    # UCB exploration (must be > 0.5)
    strategy_beta: float = 1.0      # Wide bound multiplier for Set F
    slate_size: int = 10            # Documents per result page
    grace_period: int = 500         # Rounds before elimination starts

    # --- Arm-Specific ---
    pca_dim: int = 20               # Feature reduction for LinUCB/LinearTS
    linucb_alpha: float = 1.0       # LinUCB exploration parameter
    xgb_n_estimators: int = 100     # XGBoost trees
    xgb_max_depth: int = 6          # XGBoost tree depth
```

**Key Parameters Explained:**

| Parameter | What It Controls | Trade-off |
|-----------|-----------------|-----------|
| `n_rounds` | Experiment length | More rounds → better convergence data, but slower |
| `strategy_alpha` | UCB exploration width | Higher → more exploration, slower convergence |
| `grace_period` | Protection for learning arms | Higher → LinUCB/LinearTS get more time to learn before elimination |
| `slate_size` | Results per page | Matches typical search engine (10 results) |
| `pca_dim` | Feature compression | Lower → faster LinUCB, but loses information |

### Scenarios (`config/scenarios.py`)

Defines click model configurations for testing MDB under different user behaviors:

```python
SCENARIOS = {
    "standard": {
        "click_model_type": "pbm",
        "params": {},
        "description": "Position-Based Model (realistic multi-click)"
    },
    "cascade": {
        "click_model_type": "cascade",
        "params": {"relevance_threshold": 3, "max_depth": 10},
        "description": "User clicks first relevant item, stops (sparse feedback)"
    },
    "noisy": {
        "click_model_type": "noisy",
        "params": {"noise_prob": 0.1, "false_negative_rate": 0.1},
        "description": "10% random clicks (robustness test)"
    },
}
```

**When to Use Each Scenario:**

| Scenario | User Behavior | Tests |
|----------|--------------|-------|
| `standard` | Examines multiple results, clicks several relevant items | Normal operation — does MDB converge? |
| `cascade` | Wants ONE answer, clicks first good result and leaves | Sparse feedback — can MDB learn with few clicks? |
| `noisy` | Sometimes clicks randomly or misses relevant items | Robustness — does MDB handle attribution errors? |

### How Config Connects to Code

```
run_experiment.py
    │
    ├─► Loads ExperimentConfig (defaults or CLI overrides)
    │
    ├─► get_scenario(config.scenario)
    │       → Returns click model type and parameters
    │
    ├─► Creates arms from config.arm_pool_list
    │       → Uses config.pca_dim, config.linucb_alpha, etc.
    │
    ├─► Creates MDBPolicy with config.strategy_alpha, config.grace_period
    │
    └─► Runs Simulator for config.n_rounds iterations
```

### Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (fewer rounds)
python3.11 run_experiment.py --quick

# Full experiment with defaults
python3.11 run_experiment.py

# Custom configuration via CLI
python3.11 run_experiment.py \
    --arms random xgboost linucb linear_ts \
    --scenario cascade \
    --rounds 5000
```

## References

This implementation is based on the following papers:

| Algorithm | Paper
|-----------|-------|
| **Multi-Dueling Bandits** | Brost et al., "Multi-dueling bandits with dependent arms"|
| **LinUCB** | Li et al., "A contextual-bandit approach to personalized news article recommendation"|
| **Multileaving (Team Draft)** | Schuth et al., "Multileaved comparisons for fast online evaluation" |
| **Dueling Bandits** | Yue & Joachims, "Interactively optimizing information retrieval systems as a dueling bandit" |
| **Yahoo LTR Dataset** | Chapelle & Chang, "Yahoo! learning to rank challenge overview" |
| **Historical Reuse** | Hofmann et al., "Reusing historical interaction data for faster online learning to rank for IR" |

### How Papers Map to Code

- **MDB Policy** (`src/policies/mdb.py`) — Implements Brost et al. 2016 with Set E/F and UCB bounds
- **LinUCB Arm** (`src/arms/linucb.py`) — Implements Li et al. 2010 contextual bandit with PCA
- **Linear TS Arm** (`src/arms/linear_ts.py`) — Thompson Sampling variant of contextual bandit
- **Team Draft** (`src/multileaving/team_draft.py`) — Implements Schuth et al. 2014 multileaving
- **Data** (`data/processed/yahoo_parquet/`) — Yahoo LTR Challenge dataset
