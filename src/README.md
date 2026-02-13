# Source Code (`src/`)

## Modules

| Module | Description | Key Files |
|--------|-------------|-----------|
| `agents/` | 7 bandit estimators (LinUCB → Deep Bandits) | `base_moo_agent.py` (interface), `linucb_agent.py`, `neural_ucb_agent.py` |
| `env/` | Environment simulation | `context_generator.py`, `reward_simulator.py` |
| `policy/` | 10 MOO policies in 3 families | `moo_policies.py`, `exact_moo/`, `metaheuristics/` |
| `infra/` | Real-time infrastructure clients | `kafka_messenger.py`, `redis_client.py` |
| `evaluation/` | Off-policy evaluation | `ips_evaluator.py` |
| `utils/` | Math utilities | `math_utils.py` (Sherman-Morrison) |

## Architecture

All agents implement `BaseMOOAgent`, which defines:
- `predict_all(context)` → predictions for all arms × all objectives
- `update(context, arm, rewards)` → learn from observed reward
- `select_arm(context, policy)` → choose arm using a MOO policy

This standardized interface allows any of the 7 estimators to work with any of the 10 MOO policies.
