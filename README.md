# 🎯 Multi-Objective Ad Recommendation System

A comprehensive multi-objective contextual bandit system for ad recommendation, implementing **7 bandit estimators** and **10 MOO policies** with real-time infrastructure (Kafka + Redis).

> **PFE Project** — Houbbadi Abderrahim, February 2026

---

## 📂 Project Structure

```text
pfe-recsys-ads/
├── config/
│   └── settings.py              # All hyperparameters (centralized)
├── src/
│   ├── agents/                   # 7 bandit estimators
│   │   ├── base_moo_agent.py     # Abstract multi-objective interface
│   │   ├── linucb_agent.py       # LinUCB (Sherman-Morrison)
│   │   ├── thompson_sampling_agent.py
│   │   ├── neural_ucb_agent.py   # NeuralUCB (PyTorch)
│   │   ├── neural_ts_agent.py    # Neural Thompson Sampling
│   │   ├── deep_bandit_agent.py  # Bootstrap ensemble
│   │   ├── offline_online_agent.py
│   │   └── delayed_feedback_agent.py
│   ├── env/                      # Environment simulation
│   │   ├── context_generator.py  # Synthetic user contexts
│   │   └── reward_simulator.py   # Click + Revenue simulation
│   ├── policy/                   # 10 MOO policies
│   │   ├── moo_policies.py       # Scalarization, ε-Constraint, Pareto
│   │   ├── pareto_utils.py       # Pareto dominance utilities
│   │   ├── exact_moo/            # MOBB, TwoPhase, OSS, MODP, MOA*
│   │   └── metaheuristics/       # NSGA-II, MOEA/D
│   ├── infra/                    # Real-time infrastructure
│   │   ├── kafka_messenger.py    # Kafka producer/consumer
│   │   └── redis_client.py       # Model state persistence
│   └── evaluation/
│       └── ips_evaluator.py      # Off-policy evaluation (IPS)
├── experiments/                  # Runnable benchmarks
│   ├── offline_simulation.py     # Single-agent offline test
│   ├── realtime_simulation.py    # Kafka + Redis real-time loop
│   ├── global_comparison.py      # 7×10 = 70 combination benchmark
│   ├── moo_benchmark.py          # MOO policy comparison
│   └── benchmark_large_k.py      # Scalability test (K=100)
├── tests/                        # Unit tests
├── docker-compose.yml            # Kafka + ZooKeeper + Redis
├── pyproject.toml                # Dependencies (uv)
└── requirements.txt              # pip-compatible dependencies
```

## ✨ Features

### 7 Bandit Estimators

| Estimator | Type | Exploration Strategy |
|-----------|------|---------------------|
| **LinUCB** | Linear | Upper Confidence Bound (deterministic) |
| **Thompson Sampling** | Linear/Bayesian | Posterior sampling (stochastic) |
| **NeuralUCB** | Neural | Gradient-based uncertainty |
| **Neural Thompson Sampling** | Neural | Parameter perturbation |
| **Deep Bandits** | Neural/Ensemble | Bootstrap uncertainty (5 networks) |
| **Offline-to-Online** | Hybrid | Pre-train on logs, then LinUCB online |
| **Delayed Feedback** | Linear | Handles delayed reward signals |

### 10 MOO Policies

| Family | Policies |
|--------|----------|
| **Scalarization** | Linear Scalarization, ε-Constraint, Pareto-Chebyshev |
| **Exact Methods** | MOBB, Two-Phase, OSS, MODP, MOA* |
| **Metaheuristics** | NSGA-II, MOEA/D |

### Real-Time Infrastructure

- **Apache Kafka** — Event streaming for ad requests
- **Redis** — Persistent model state (A⁻¹, b matrices)
- **Docker Compose** — One-command infrastructure setup

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
# or with uv:
uv sync

# 2. Run the 70-combination global benchmark
python experiments/global_comparison.py

# 3. Run with K=100 arms (scalability test)
python experiments/global_comparison.py --k 100

# 4. (Optional) Start real-time infrastructure
docker compose up -d
python experiments/realtime_simulation.py
```

## 📊 Key Results

### K=5 Arms (500 iterations)

| Estimator | Best Policy | CTR | Revenue |
|-----------|-------------|-----|---------|
| **Thompson Sampling** | **NSGA-II** | **0.584** | **0.217** |
| LinUCB | NSGA-II | 0.534 | 0.210 |
| DelayedFB | Scalar | 0.566 | 0.203 |

### K=100 Arms (Scalability)

| Estimator | Best Policy | CTR | Revenue |
|-----------|-------------|-----|---------|
| **LinUCB** | **NSGA-II** | **0.556** | **0.227** |
| Thompson | MODP | 0.542 | 0.216 |
| DelayedFB | MOA* | 0.544 | 0.225 |

### Key Findings

- **Thompson Sampling** dominates for small K (Bayesian exploration adapts naturally)
- **LinUCB** scales better to large K (data-efficient linear model)
- **NSGA-II** is the most robust MOO policy across both scenarios
- Execution time varies 800× between LinUCB (~300ms) and NeuralUCB (~24s) at K=100

## 🧪 Tests

```bash
pytest tests/ -v
```

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Deep Learning | PyTorch (CUDA) |
| Message Queue | Apache Kafka (confluent-7.4) |
| State Store | Redis 7.0 |
| Orchestration | Docker Compose |
| Package Manager | uv / pip |

## 📄 License

Academic project — PFE 2026.
