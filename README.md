# 🎯 Multi-Objective Semantic Ad Recommendation System

A contextual bandit system for ad recommendation research, comparing **16 agents** across **10 MOO policies** in a semantic cold-start environment.

> **PFE Project** — Houbbadi Abderrahim, 2026

---

## 📂 Project Structure

```text
pfe-recsys-ads/
│
├── experiments/                    # ← Main benchmark scripts
│   ├── mega_semantic_comparison.py # Main benchmark (16 agents × 10 policies)
│   ├── best_across_policies.py     # Best-policy-per-agent analysis
│   ├── regenerate_best_plots.py    # Regenerate trajectory & Pareto plots
│   └── zero_shot_demo.py           # Zero-shot transfer demonstration
│
├── src/
│   ├── agents/                     # 16 agent implementations
│   │   ├── base_agent.py           # Abstract base class
│   │   ├── base_moo_agent.py       # MOO interface
│   │   ├── linucb_agent.py         # Classical LinUCB
│   │   ├── thompson_sampling_agent.py
│   │   ├── neural_ucb_agent.py     # NeuralUCB (PyTorch)
│   │   ├── neural_ts_agent.py      # NeuralTS
│   │   ├── deep_bandit_agent.py    # DeepBandit
│   │   ├── offline_online_agent.py # Offline2Online
│   │   ├── delayed_feedback_agent.py
│   │   ├── global_semantic_linucb.py   # H-LinUCB (hybrid)
│   │   ├── global_semantic_neural.py   # H-NeuralUCB, H-NeuralTS, H-DeepBandit
│   │   ├── global_semantic_others.py   # H-Offline2On, H-DelayedFB, H-Thompson
│   │   ├── multi_obj_agent.py
│   │   └── llm_agents/             # LlamaReasoning, LlamaInstruct
│   │
│   ├── policy/                     # 10 MOO policies
│   │   ├── moo_policies.py         # Scalar, ε-Constraint, Pareto-Ch
│   │   ├── exact_moo/              # MOBB, TwoPhase, OSS, MODP, MOA*
│   │   └── metaheuristics/         # NSGA-II, MOEA/D
│   │
│   ├── env/
│   │   ├── semantic_env/           # SemanticRewardSimulator, TextDatasetLoader
│   │   ├── reward_simulator.py
│   │   └── context_generator.py
│   │
│   ├── llm/                        # SentenceTransformer, Ollama, Gemini clients
│   └── utils/                      # Math utilities (Sherman-Morrison, etc.)
│
├── metrics/                        # ← Final benchmark results
│   ├── eng_matrix.csv              # 16×10 engagement scores
│   ├── rev_matrix.csv              # 16×10 revenue scores
│   ├── time_matrix.csv             # 16×10 execution times
│   ├── trajectories_best_policy.json  # Per-iteration data (best policy/agent)
│   ├── zero_shot_transfer_demo.png
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
├── data/                           # Raw datasets
├── config/                         # Hyperparameters and env settings
│
├── archive/                        # ← Previous project phases
│   ├── phase1_realworld_offline/   # IPS & Rejection Sampling on Criteo/OBD
│   ├── phase2_infrastructure/      # Redis + Kafka real-time pipeline
│   ├── phase3_autoencoders/        # VAE-based representation learning
│   ├── phase4_early_benchmarks/    # Superseded benchmark scripts
│   ├── phase5_scripts_debug/       # Test & debug utilities
│   └── old_results_and_plots/      # Plots from phases 1–4
│
└── documentation/                  # LaTeX reports & presentations
    ├── pfe-presentation/
    ├── report/
    ├── docs/
    └── pfe-report/
```

---

## 🚀 Running the Benchmark

```bash
# Run the full 16-agent × 10-policy benchmark
uv run python experiments/mega_semantic_comparison.py

# Re-generate all plots using best policy per agent (requires saved CSV matrices)
uv run python experiments/regenerate_best_plots.py

# Generate best-vs-best comparison tables and bar charts
uv run python experiments/best_across_policies.py

# Generate the focused zero-shot transfer demonstration
uv run python experiments/zero_shot_demo.py
```

---

## 🏆 Key Results

| Metric | Winner | Score | Policy |
|:---|:---|:---:|:---|
| Best Engagement | H-DeepBandit | 0.7532 | Pareto-Ch |
| Best Revenue | H-DeepBandit | **0.0944** | ε-Constraint |
| Fastest Recovery | H-LinUCB | Δ=0.132 | OSS |
| Zero-Shot Gap | H-LinUCB | +0.132 | vs LinUCB (same policy) |

- **5/7** hybrid agents outperform classical on engagement (best vs best)
- **6/7** hybrid agents outperform classical on revenue (best vs best)
- **Thompson** and **NeuralTS** are exceptions: stochastic posterior sampling conflicts with global model smoothing

---

## 🧩 Agent Architecture

| Family | Agents | Description |
|:---|:---|:---|
| **Classical** | LinUCB, Thompson, NeuralUCB, NeuralTS, DeepBandit, Offline2On, DelayedFB | Disjoint per-arm models |
| **Hybrid** | H-LinUCB, H-Thompson, H-NeuralUCB, H-NeuralTS, H-DeepBandit, H-Offline2On, H-DelayedFB | Global semantic model via `all-MiniLM-v2` |
| **LLM** | LlamaReasoning, LlamaInstruct | Local Ollama inference |

---

## 📦 Archive Index

| Folder | Phase | Key Technology |
|:---|:---|:---|
| `archive/phase1_realworld_offline/` | 1 | IPS, Rejection Sampling, Criteo dataset |
| `archive/phase2_infrastructure/` | 2 | Redis, Kafka, real-time loop |
| `archive/phase3_autoencoders/` | 3 | VAE, PyTorch representation learning |
| `archive/phase4_early_benchmarks/` | 4 | Preliminary bandit comparisons |
| `archive/phase5_scripts_debug/` | 5 | Data download, connection tests |
| `archive/old_results_and_plots/` | All | Historical plots and results |
