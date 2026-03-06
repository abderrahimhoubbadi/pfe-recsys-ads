# 🎯 Closed-Loop Real-Time Recommender System for Google Ads

A multi-objective contextual bandit system for ad recommendation, featuring **16 agents**, **10 MOO policies**, and a **real-time deployment pipeline** with Docker and GCP support.

> **PFE Project** — Houbbadi Abderrahim, Devoteam Maroc, 2026

---

## 📂 Project Structure

```text
pfe-recsys-ads/
│
├── src/
│   ├── agents/                         # 16 agent implementations
│   │   ├── linucb_agent.py             # Classical LinUCB
│   │   ├── thompson_sampling_agent.py  # Thompson Sampling
│   │   ├── neural_ucb_agent.py         # NeuralUCB (PyTorch)
│   │   ├── neural_ts_agent.py          # NeuralTS
│   │   ├── deep_bandit_agent.py        # DeepBandit
│   │   ├── offline_online_agent.py     # Offline2Online
│   │   ├── delayed_feedback_agent.py   # DelayedFB
│   │   ├── global_semantic_linucb.py   # H-LinUCB (hybrid)
│   │   ├── global_semantic_neural.py   # H-NeuralUCB, H-NeuralTS, H-DeepBandit
│   │   ├── global_semantic_others.py   # H-Offline2On, H-DelayedFB, H-Thompson
│   │   └── llm_agents/                 # LlamaReasoning, LlamaInstruct
│   │
│   ├── policy/                         # 10 MOO policies
│   │   ├── moo_policies.py             # Scalar, ε-Constraint, Pareto-Chebyshev
│   │   ├── exact_moo/                  # MOBB, TwoPhase, OSS, MODP, MOA*
│   │   └── metaheuristics/             # NSGA-II, MOEA/D
│   │
│   ├── api/                            # ← FastAPI Real-Time Service
│   │   ├── main.py                     # App entry (uvicorn)
│   │   ├── recommendation_service.py   # H-DeepBandit service layer
│   │   └── schemas.py                  # Pydantic request/response models
│   │
│   ├── infra/                          # ← Infrastructure (Docker / GCP)
│   │   ├── factory.py                  # Abstract interfaces + factory pattern
│   │   ├── redis_client.py             # State persistence (Redis / Memorystore)
│   │   ├── kafka_messenger.py          # Kafka adapters (Option A)
│   │   └── pubsub_client.py            # GCP Pub/Sub adapters (Option B)
│   │
│   ├── streaming/                      # ← Kafka/Pub/Sub closed-loop consumer
│   │   └── consumer.py                 # Impression → Decision → Feedback loop
│   │
│   ├── env/semantic_env/               # SemanticRewardSimulator
│   ├── llm/                            # SentenceTransformer, Ollama, Gemini
│   └── utils/                          # Math utilities (Sherman-Morrison)
│
├── experiments/                        # Benchmark scripts
│   ├── mega_semantic_comparison.py     # Main 16×10 benchmark
│   ├── best_across_policies.py         # Best-vs-Best analysis
│   ├── regenerate_best_plots.py        # Trajectory & Pareto plots
│   ├── zero_shot_demo.py              # Zero-shot transfer demo
│   └── generate_delta_joint_plot.py    # Joint hybridization delta
│
├── tests/
│   └── test_integration.py             # End-to-end pipeline test
│
├── metrics/                            # Final benchmark results (CSV + PNG)
├── deploy/gcp_deploy.sh                # GCP Cloud Run deployment script
├── Dockerfile                          # Container image
├── docker-compose.yml                  # 5-service local stack
└── documentation/                      # Reports & presentations
```

---

## 🚀 Quick Start

### 1. Run the Integration Test

```bash
uv run python -m tests.test_integration
```

### 2. Start the API Server

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Make a Recommendation

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

### 4. Deploy with Docker Compose

```bash
docker compose up -d    # Starts API + Redis + Kafka + Grafana
```

### 5. Deploy to GCP (Cloud Run)

```bash
export GCP_PROJECT_ID=your-project-id
bash deploy/gcp_deploy.sh
```

---

## 🏆 Benchmark Results

| Metric | Winner | Score | Policy |
|:---|:---|:---:|:---|
| Best Engagement | H-DeepBandit | 0.7532 | Pareto-Ch |
| Best Revenue | H-DeepBandit | **0.0944** | ε-Constraint |
| Zero-Shot Gap | H-LinUCB | +0.132 | vs LinUCB |

### Impact of Semantic Hybridization (Joint Delta Analysis)

- **1/7 Strict Dominance (Win-Win)**: `H-DeepBandit × ε-Constraint` improves BOTH engagement (+0.070) and revenue (+0.012).
- **5/7 Favorable Trade-offs**: Most hybrid agents improve revenue (up to +0.014) with a slight engagement drop.
- **1/7 Classical Wins**: `H-Thompson` degrades — global model smoothing interferes with stochastic posterior sampling.

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
| **Hybrid** | H-LinUCB, H-Thompson, H-NeuralUCB, H-NeuralTS, H-DeepBandit, H-Offline2On, H-DelayedFB | Global semantic model `[x_user ‖ x_ad]` |
| **LLM** | LlamaReasoning, LlamaInstruct | Zero-shot oracles (Ollama) |
