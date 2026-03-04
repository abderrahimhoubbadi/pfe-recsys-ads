# 🎯 Multi-Objective Semantic Ad Recommendation System

A contextual bandit system for ad recommendation research, comparing **16 agents** across **10 MOO policies** in a semantic cold-start environment.

> **PFE Project** — Houbbadi Abderrahim, 2026

---

## 📂 Project Structure

```text
pfe-recsys-ads/
│
├── experiments/                        # ← Benchmark scripts
│   ├── mega_semantic_comparison.py     # Main benchmark (16 agents × 10 policies)
│   ├── best_across_policies.py         # Best-policy-per-agent analysis
│   ├── regenerate_best_plots.py        # Regenerate trajectory & Pareto plots
│   ├── zero_shot_demo.py               # Zero-shot transfer demonstration
│   └── generate_delta_joint_plot.py    # Joint hybridisation delta plotting
│
├── src/
│   ├── agents/                         # 16 agent implementations
│   │   ├── base_agent.py               # Abstract base class
│   │   ├── base_moo_agent.py           # MOO interface
│   │   ├── linucb_agent.py             # Classical LinUCB
│   │   ├── thompson_sampling_agent.py
│   │   ├── neural_ucb_agent.py         # NeuralUCB (PyTorch)
│   │   ├── neural_ts_agent.py          # NeuralTS
│   │   ├── deep_bandit_agent.py        # DeepBandit
│   │   ├── offline_online_agent.py     # Offline2Online
│   │   ├── delayed_feedback_agent.py
│   │   ├── global_semantic_linucb.py   # H-LinUCB (hybrid)
│   │   ├── global_semantic_neural.py   # H-NeuralUCB, H-NeuralTS, H-DeepBandit
│   │   ├── global_semantic_others.py   # H-Offline2On, H-DelayedFB, H-Thompson
│   │   ├── multi_obj_agent.py
│   │   └── llm_agents/                 # LlamaReasoning, LlamaInstruct
│   │
│   ├── policy/                         # 10 MOO policies
│   │   ├── moo_policies.py             # Scalar, ε-Constraint, Pareto-Ch
│   │   ├── exact_moo/                  # MOBB, TwoPhase, OSS, MODP, MOA*
│   │   └── metaheuristics/             # NSGA-II, MOEA/D
│   │
│   ├── env/
│   │   ├── semantic_env/               # SemanticRewardSimulator, TextDatasetLoader
│   │   ├── reward_simulator.py
│   │   └── context_generator.py
│   │
│   ├── llm/                            # SentenceTransformer, Ollama, Gemini clients
│   └── utils/                          # Math utilities (Sherman-Morrison, etc.)
│
├── metrics/                            # ← Final benchmark results
│   ├── eng_matrix.csv                  # 16×10 engagement scores
│   ├── rev_matrix.csv                  # 16×10 revenue scores
│   ├── time_matrix.csv                 # 16×10 execution times
│   ├── trajectories_best_policy.json   # Per-iteration data (best policy/agent)
│   ├── zero_shot_transfer_demo.png
│   ├── hybridization_delta_joint.png
│   ├── best_across_policies_bars.png
│   ├── hybridization_delta_best_vs_best.png
│   ├── trajectory_cumulative_engagement.png
│   ├── trajectory_post_shock_recovery.png
│   ├── mega_pareto.png
│   ├── best_of_class_pareto.png
│   ├── mega_heatmap_engagement.png
│   ├── mega_heatmap_revenue.png
│   ├── mega_heatmap_time.png
│   └── radar_capabilities.png
│
├── data/                               # Raw datasets
├── config/                             # Hyperparameters and env settings
└── documentation/                      # LaTeX reports & presentations
```

---

## 🚀 Running the Benchmark

This project provides a comprehensive pipeline to simulate and evaluate the performance of contextual bandits on multi-objective ad recommendation tasks.

```bash
# 1. Run the full 16-agent × 10-policy benchmark
uv run python experiments/mega_semantic_comparison.py

# 2. Re-generate all plots using best policy per agent (requires saved CSV matrices)
uv run python experiments/regenerate_best_plots.py

# 3. Generate best-vs-best comparison tables and bar charts
uv run python experiments/best_across_policies.py

# 4. Generate the focused zero-shot transfer demonstration
uv run python experiments/zero_shot_demo.py

# 5. Generate the joint 2D scatter plot of hybridization impacts
uv run python experiments/generate_delta_joint_plot.py
```

---

## 🏆 Key Results & Benchmarking Walkthrough

The benchmarking evaluates the agents on their ability to optimize both **Engagement (CTR)** and **Revenue (eCPM)**, particularly when faced with a "Cold-Start Shock" (the sudden introduction of new ads into the system).

| Metric | Winner | Score | Policy |
|:---|:---|:---:|:---|
| Best Engagement | H-DeepBandit | 0.7532 | Pareto-Ch |
| Best Revenue | H-DeepBandit | **0.0944** | ε-Constraint |
| Fastest Recovery | H-LinUCB | Δ=0.132 | OSS |
| Zero-Shot Gap | H-LinUCB | +0.132 | vs LinUCB (same policy) |

### Impact of Semantic Hybridization

The core hypothesis is that **hybrid agents**—which use a global model operating on combined user+ad semantic embeddings (`all-MiniLM-v2`)—can generalize to unseen ads via zero-shot transfer, overcoming the cold-start problem that paralyzes standard disjoint bandits.

Comparing the *best achievable performance* (Best-vs-Best across all 10 MOO policies) of hybrid models vs. their classical counterparts reveals the following:

- **1/7 Strict Dominance (Win-Win)**: `H-DeepBandit × ε-Constraint` improves BOTH maximum engagement (+0.070) and maximum revenue (+0.012).
- **5/7 Favorable Trade-offs**: Most hybrid agents (`H-LinUCB`, `H-NeuralUCB`, `H-Offline2On`, `H-DelayedFB`) significantly improve revenue (up to +0.014) with only a slight engagement drop.
- **1/7 Classical Wins (Lose-Lose)**: `H-Thompson` degrades under hybridization, as the global model's parameter smoothing interferes with the stochastic posterior sampling needed for exploration.

---

## 🧩 Agent Architecture Overview

| Family | Agents | Description |
|:---|:---|:---|
| **Classical** | LinUCB, Thompson, NeuralUCB, NeuralTS, DeepBandit, Offline2On, DelayedFB | Standard disjoint approach (one isolated model learned per arm). Suffers heavily from cold-starts. |
| **Hybrid** | H-LinUCB, H-Thompson, H-NeuralUCB, H-NeuralTS, H-DeepBandit, H-Offline2On, H-DelayedFB | Global single model taking `[x_user || x_ad]` embeddings. Achieves semantic zero-shot transfer. |
| **LLM** | LlamaReasoning, LlamaInstruct | Pure Zero-Shot Oracles using local Ollama inference. Establishes the purely semantic performance ceiling without active learning. |
