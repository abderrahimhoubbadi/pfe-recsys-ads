# 🎯 Real-Time Ad Recommendation System with Semantic Contextual Bandits

A **multi-objective contextual bandit** system for real-time ad recommendation, featuring **16 agents**, **10 MOO policies**, and a **closed-loop deployment pipeline** with Docker and GCP support.

> **PFE (Projet de Fin d'Études)** — EMI, Université Mohammed V de Rabat, 2025/2026
> **Auteur** : Houbbadi Abderrahim
> **Entreprise** : Devoteam Maroc — BU Data Driven
> **Encadrant professionnel** : Fahd Idrissi Khamlichi

---

## 🔬 About

This project addresses the problem of **real-time ad selection** in programmatic advertising (RTB), formulated as a **Multi-Objective Integer Program (MOIP)**. The system must simultaneously:
- **Maximize engagement** (CTR) and **revenue** (eCPM) — conflicting objectives
- **Resolve cold-start** — new ads have zero interaction history
- **Respect latency** — decisions must be made in **< 50 ms**

Our key contribution is a **Hybrid Semantic Contextual Bandit** architecture that uses **SentenceTransformer** embeddings to enable **zero-shot transfer** for new ads, eliminating the cold-start exploration cost.

---

## 📂 Project Structure

```text
pfe-recsys-ads/
│
├── src/
│   ├── agents/                         # 16 agent implementations
│   │   ├── linucb_agent.py             # Classical LinUCB (UCB exploration)
│   │   ├── thompson_sampling_agent.py  # Thompson Sampling (Bayesian)
│   │   ├── neural_ucb_agent.py         # NeuralUCB (gradient-based UCB)
│   │   ├── neural_ts_agent.py          # NeuralTS (neural Thompson)
│   │   ├── deep_bandit_agent.py        # DeepBandit (ε-greedy ensemble)
│   │   ├── offline_online_agent.py     # Offline2Online transfer
│   │   ├── delayed_feedback_agent.py   # Delayed feedback handling
│   │   ├── global_semantic_linucb.py   # H-LinUCB (hybrid semantic)
│   │   ├── global_semantic_neural.py   # H-NeuralUCB, H-NeuralTS, H-DeepBandit
│   │   ├── global_semantic_others.py   # H-Offline2On, H-DelayedFB, H-Thompson
│   │   └── llm_agents/                 # LlamaReasoning, LlamaInstruct
│   │
│   ├── policy/                         # 10 MOO policies
│   │   ├── moo_policies.py             # Scalar, ε-Constraint, Pareto-Chebyshev
│   │   ├── exact_moo/                  # MOBB, TwoPhase, OSS, MODP, MOA*
│   │   └── metaheuristics/             # NSGA-II, MOEA/D
│   │
│   ├── api/                            # FastAPI real-time service
│   │   ├── main.py                     # App entry (uvicorn)
│   │   ├── recommendation_service.py   # H-DeepBandit service layer
│   │   └── schemas.py                  # Pydantic request/response models
│   │
│   ├── infra/                          # Infrastructure (Docker / GCP)
│   │   ├── factory.py                  # Abstract interfaces + factory pattern
│   │   ├── redis_client.py             # State persistence (Redis / Memorystore)
│   │   ├── kafka_messenger.py          # Kafka adapters (local)
│   │   └── pubsub_client.py            # GCP Pub/Sub adapters (cloud)
│   │
│   ├── streaming/                      # Kafka/Pub/Sub closed-loop consumer
│   │   └── consumer.py                 # Impression → Decision → Feedback loop
│   │
│   ├── env/semantic_env/               # Semantic reward simulator
│   │   ├── semantic_reward_simulator.py
│   │   └── text_dataset_loader.py      # 60 B2B ads + 200 user profiles
│   │
│   ├── evaluation/                     # Evaluation utilities
│   ├── llm/                            # SentenceTransformer, Ollama, Gemini clients
│   └── utils/                          # Math utilities (Sherman-Morrison)
│
├── experiments/                        # Benchmark scripts
│   ├── mega_semantic_comparison.py     # Main 16×10 benchmark (142 configs)
│   ├── best_across_policies.py         # Best-vs-Best analysis
│   ├── regenerate_best_plots.py        # Trajectory & Pareto plots
│   ├── zero_shot_demo.py              # Zero-shot transfer demonstration
│   ├── generate_delta_joint_plot.py    # Joint hybridization delta
│   └── offline_obd_validation.py       # OBD offline validation (SNIPS)
│
├── tests/
│   ├── test_integration.py             # End-to-end pipeline test
│   ├── test_load.py                    # Load/stress test (latency)
│   └── test_realtime_simulation.py     # Real-time simulation (500 iters)
│
├── scripts/
│   ├── download_obd.py                 # Download Open Bandit Dataset
│   └── extract_bts_npz.py             # Extract BTS feedback data
│
├── config/settings.py                  # Centralized configuration
├── metrics/                            # Benchmark results (CSV + PNG)
├── deploy/gcp_deploy.sh                # GCP Cloud Run deployment script
├── Dockerfile                          # Container image
├── docker-compose.yml                  # 5-service local stack
└── pyproject.toml                      # Dependencies (uv)
```

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### 1. Install Dependencies

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 2. Run the Integration Test

```bash
uv run python -m tests.test_integration
```

### 3. Start the API Server

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Make a Recommendation

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "user_text": "homme 28 ans passionné de technologie",
    "available_ads": [
      {"ad_id": 1, "title": "RTX 5090", "description": "GPU gaming", "category": "tech"},
      {"ad_id": 2, "title": "Cours IA", "description": "Formation deep learning", "category": "éducation"}
    ]
  }'
```

### 5. Deploy with Docker Compose

```bash
docker compose up -d    # Starts API + Redis + Kafka + Grafana
```

### 6. Deploy to GCP (Cloud Run)

```bash
export GCP_PROJECT_ID=your-project-id
bash deploy/gcp_deploy.sh
```

---

## 🏆 Benchmark Results (142 Configurations)

### Champion: H-DeepBandit × ε-Constraint

| Metric | Score | Rank |
|:---|:---:|:---|
| Engagement (CTR) | **0.753** | Top 5 / 142 |
| Revenue (eCPM) | **0.094** | **#1** / 142 |
| Zero-Shot Gap | +0.132 | vs classical LinUCB |
| OBD Validation | **+45.5%** | vs production BTS policy |

### Impact of Semantic Hybridization

- **5/7 Win-Win**: Hybrid agents simultaneously improve engagement AND revenue
- **Best gain**: H-DeepBandit × ε-Constraint — engagement +0.070, revenue +0.012
- **Exceptions**: Thompson Sampling and NeuralTS — stochastic exploration conflicts with high-dimensional semantic space

### Offline Validation (Open Bandit Dataset)

Validated on **26M real events** from ZOZO using Inverse Propensity Scoring (SNIPS):

| Policy | CTR (SNIPS) |
|:---|:---:|
| Random (floor) | 0.42% |
| Bernoulli TS (production) | 0.77% |
| **H-DeepBandit (ours)** | **1.12%** |

---

## 🏗️ Architecture — Closed-Loop Pipeline

```
                          ┌───────────────────────┐
  Impression Event ──▷    │  SentenceTransformer  │  ──▷  x_u ∈ R^384
                          │   (all-MiniLM-L6-v2)  │
                          └───────────────────────┘
                                     │
                     ┌───────────────▼───────────────┐
                     │      H-DeepBandit Agent       │
                     │  (5-ensemble global network)  │
                     │    context = [x_u ‖ x_ad]     │
                     └───────────────┬───────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  ε-Constraint MOO   │
                          │  max CTR s.t. Rev≥ε │
                          └──────────┬──────────┘
                                     │
                           selected_ad_id ──▷ User
                                     │
                              Click/Conv ──▷ agent.update()  ← Closed Loop
```

### Deployment Options

| Component | Option A: Docker | Option B: GCP |
|:---|:---|:---|
| API | FastAPI container | Cloud Run |
| Queue | Apache Kafka | Pub/Sub |
| State | Redis container | Memorystore |
| Monitoring | Grafana | Cloud Monitoring |

---

## 🧩 Agent Families

| Family | Agents | Architecture |
|:---|:---|:---|
| **Classical** | LinUCB, Thompson, NeuralUCB, NeuralTS, DeepBandit, Offline2On, DelayedFB | Disjoint per-arm models |
| **Hybrid** | H-LinUCB, H-Thompson, H-NeuralUCB, H-NeuralTS, H-DeepBandit, H-Offline2On, H-DelayedFB | Global semantic model `[x_user ‖ x_ad]` ∈ ℝ⁷⁶⁸ |
| **LLM** | LlamaReasoning, LlamaInstruct | Zero-shot semantic oracles (Ollama) |

---

## 📊 Key Performance Metrics

| Metric | Value |
|:---|:---:|
| Decision latency (avg) | **12 ms** |
| Decision latency (P95) | **15 ms** |
| Throughput | **84 req/s** (single-thread) |
| Cold-start recovery | **Instant** (zero-shot) |
| Benchmark configs tested | **142** |

---

## 📄 License

This project was developed as a **Projet de Fin d'Études (PFE)** at [EMI](https://www.emi.ac.ma/) — École Mohammadia d'Ingénieurs, Université Mohammed V de Rabat, in partnership with [Devoteam Maroc](https://www.devoteam.com/).

---

## 🙏 Acknowledgements

- **Devoteam Maroc** — BU Data Driven, for hosting this research project
- **EMI** — École Mohammadia d'Ingénieurs, for the academic framework
- **Fahd Idrissi Khamlichi** — Professional supervisor at Devoteam
