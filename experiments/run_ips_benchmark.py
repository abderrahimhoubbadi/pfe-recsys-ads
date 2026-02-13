"""
IPS Benchmark with Multi-Objective Policies.

Compares: Random vs Scalarization vs ε-Constraint vs Pareto using Off-Policy Evaluation.
"""
import sys
import os
import json
import numpy as np
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import DIMENSION, N_ARMS, EPSILON_THRESHOLD
from src.evaluation.ips_evaluator import IPSEvaluator
from src.agents.multi_obj_agent import MultiObjectiveLinUCBAgent
from src.agents.base_agent import BaseAgent
from src.policy.moo_policies import (
    linear_scalarization_policy,
    epsilon_constraint_policy,
    pareto_frontier_policy
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("IPSBenchmark")


class RandomAgent(BaseAgent):
    """Simple random agent for baseline."""
    def __init__(self, n_arms, dimension):
        super().__init__(n_arms, dimension)
        self.rng = np.random.default_rng(42)
        
    def select_arm(self, context):
        return self.rng.integers(0, self.n_arms)
        
    def update(self, context, arm, reward):
        pass
        
    def get_model_params(self):
        return {}


class MOOAgentWrapper:
    """Wrapper to make MOO agent work with IPS evaluator."""
    def __init__(self, agent, policy_fn):
        self.agent = agent
        self.policy_fn = policy_fn
        self.n_arms = agent.n_arms
        
    def select_arm(self, context):
        return self.agent.select_arm(context, policy=self.policy_fn)
    
    def update(self, context, arm, reward):
        # Handle both single value and dict rewards
        if isinstance(reward, dict):
            self.agent.update(context, arm, reward)
        else:
            self.agent.update(context, arm, {'click': reward, 'revenue': reward * 0.5})


def load_logs(filepath):
    logger.info(f"Loading logs from {filepath}...")
    dataset = []
    with open(filepath, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    logger.info(f"Loaded {len(dataset)} events.")
    return dataset


def run_benchmark():
    LOG_FILE = "data/logs_10k.jsonl"
    M_CAP = 10.0
    
    if not os.path.exists(LOG_FILE):
        logger.error(f"File {LOG_FILE} not found. Generate logs first.")
        return
        
    dataset = load_logs(LOG_FILE)
    
    # 1. Random Policy (Baseline)
    random_agent = RandomAgent(n_arms=N_ARMS, dimension=DIMENSION)
    ips_random = IPSEvaluator(random_agent, cap_M=M_CAP)
    score_rnd, metrics_rnd = ips_random.evaluate(dataset)
    
    # 2. Logging Policy Score
    mean_reward_logging = np.mean([d['reward'] for d in dataset])
    
    # 3. MOO Policies
    policies = [
        ("Scalarization (0.5/0.5)", linear_scalarization_policy({'click': 0.5, 'revenue': 0.5})),
        ("ε-Constraint", epsilon_constraint_policy('click', 'revenue', epsilon=EPSILON_THRESHOLD)),
        ("Pareto (Chebyshev)", pareto_frontier_policy()),
    ]
    
    moo_results = []
    for name, policy_fn in policies:
        agent = MultiObjectiveLinUCBAgent(
            n_arms=N_ARMS, 
            dimension=DIMENSION, 
            alpha=0.5,
            objectives=['click', 'revenue']
        )
        wrapped = MOOAgentWrapper(agent, policy_fn)
        evaluator = IPSEvaluator(wrapped, cap_M=M_CAP)
        score, metrics = evaluator.evaluate(dataset)
        moo_results.append((name, score, metrics))
    
    # Results
    print("\n" + "=" * 55)
    print("         IPS BENCHMARK - MOO POLICIES         ")
    print("=" * 55)
    print(f"Dataset Size:   {len(dataset)}")
    print(f"Capping M:      {M_CAP}")
    print("-" * 55)
    print(f"{'Policy':<30} {'IPS Score':>12} {'Match Rate':>10}")
    print("-" * 55)
    print(f"{'1. Random (Baseline)':<30} {score_rnd:>12.4f} {metrics_rnd['match_rate']:>9.1%}")
    print(f"{'2. Logging Policy':<30} {mean_reward_logging:>12.4f} {'100.0%':>10}")
    
    for name, score, metrics in moo_results:
        print(f"{'3. ' + name:<30} {score:>12.4f} {metrics['match_rate']:>9.1%}")
    
    print("=" * 55)
    
    # Determine winner
    best_moo = max(moo_results, key=lambda x: x[1])
    if best_moo[1] > mean_reward_logging:
        print(f"\n✅ WINNER: {best_moo[0]} beats Logging Policy!")
    else:
        print(f"\n⚠️  Best MOO ({best_moo[0]}) does not beat Logging Policy yet.")


if __name__ == "__main__":
    run_benchmark()
