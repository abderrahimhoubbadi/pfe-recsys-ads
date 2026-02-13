"""
Multi-Objective Optimization Benchmark.

Compares 10 MOO decision policies:
- 3 Scalarization methods
- 5 Exact direct methods
- 2 Metaheuristics
"""
import sys
import os
import numpy as np
import logging
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import (
    N_ITERATIONS, DIMENSION, N_ARMS, ALPHA, 
    EPSILON_THRESHOLD, OBJECTIVES
)
from src.env.context_generator import ContextGenerator
from src.env.reward_simulator import RewardSimulator
from src.agents.multi_obj_agent import MultiObjectiveLinUCBAgent

# Scalarization policies
from src.policy.moo_policies import (
    linear_scalarization_policy,
    epsilon_constraint_policy,
    pareto_frontier_policy
)

# Exact MOO policies
from src.policy.exact_moo import (
    mobb_policy,
    two_phase_policy,
    oss_policy,
    modp_policy,
    moa_star_policy
)

# Metaheuristic policies
from src.policy.metaheuristics import (
    nsga2_policy,
    moead_policy
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MOOBenchmark")


def run_experiment(policy_name: str, policy_fn, n_iter: int) -> dict:
    """
    Run a single experiment with a specific policy.
    """
    ctx_gen = ContextGenerator(dimension=DIMENSION)
    env = RewardSimulator(dimension=DIMENSION, n_arms=N_ARMS)
    agent = MultiObjectiveLinUCBAgent(
        n_arms=N_ARMS,
        dimension=DIMENSION,
        alpha=ALPHA,
        objectives=['click', 'revenue']
    )
    
    total_clicks = 0
    total_revenue = 0.0
    constraint_violations = 0
    
    start_time = time.time()
    
    for t in range(n_iter):
        context = ctx_gen.get_context()
        chosen_arm = agent.select_arm(context, policy=policy_fn)
        rewards = env.get_reward(context, chosen_arm)
        agent.update(context, chosen_arm, rewards)
        
        total_clicks += rewards['click']
        total_revenue += rewards['revenue']
        
        expected = env.get_expected_reward(context, chosen_arm)
        if expected['revenue'] < EPSILON_THRESHOLD:
            constraint_violations += 1
    
    duration = time.time() - start_time
    
    return {
        'policy': policy_name,
        'avg_ctr': total_clicks / n_iter,
        'avg_revenue': total_revenue / n_iter,
        'constraint_violation_rate': constraint_violations / n_iter,
        'time_ms': duration * 1000
    }


def main():
    logger.info("=" * 70)
    logger.info("Multi-Objective LinUCB Benchmark (10 Policies)")
    logger.info("=" * 70)
    
    # All 10 policies organized by category
    policies = [
        # Category 1: Scalarization (3)
        ("Scalarization (0.5/0.5)", linear_scalarization_policy({'click': 0.5, 'revenue': 0.5})),
        ("Epsilon-Constraint", epsilon_constraint_policy('click', 'revenue', epsilon=EPSILON_THRESHOLD)),
        ("Pareto Chebyshev", pareto_frontier_policy()),
        
        # Category 2: Exact Direct (5)
        ("MOBB", mobb_policy()),
        ("Two-Phase", two_phase_policy()),
        ("OSS (Aneja-Nair)", oss_policy()),
        ("MODP", modp_policy()),
        ("MOA*", moa_star_policy()),
        
        # Category 3: Metaheuristics (2)
        ("NSGA-II", nsga2_policy()),
        ("MOEA/D", moead_policy()),
    ]
    
    results = []
    for name, policy_fn in policies:
        logger.info(f"\nRunning: {name}...")
        result = run_experiment(name, policy_fn, N_ITERATIONS)
        results.append(result)
        logger.info(f"  CTR: {result['avg_ctr']:.4f} | Revenue: {result['avg_revenue']:.4f} | "
                   f"Violations: {result['constraint_violation_rate']:.2%} | Time: {result['time_ms']:.1f}ms")
    
    # Summary Table
    logger.info("\n" + "=" * 85)
    logger.info("SUMMARY - All 10 Policies")
    logger.info("=" * 85)
    logger.info(f"{'Policy':<25} {'CTR':>8} {'Revenue':>10} {'Violations':>12} {'Time(ms)':>10}")
    logger.info("-" * 85)
    
    for r in results:
        logger.info(f"{r['policy']:<25} {r['avg_ctr']:>8.4f} {r['avg_revenue']:>10.4f} "
                   f"{r['constraint_violation_rate']:>11.2%} {r['time_ms']:>10.1f}")
    
    # Find best policies
    logger.info("\n" + "=" * 85)
    logger.info("BEST POLICIES BY CATEGORY")
    logger.info("=" * 85)
    
    best_ctr = max(results, key=lambda x: x['avg_ctr'])
    best_rev = max(results, key=lambda x: x['avg_revenue'])
    best_constraint = min(results, key=lambda x: x['constraint_violation_rate'])
    fastest = min(results, key=lambda x: x['time_ms'])
    
    logger.info(f"Best CTR:        {best_ctr['policy']} ({best_ctr['avg_ctr']:.4f})")
    logger.info(f"Best Revenue:    {best_rev['policy']} ({best_rev['avg_revenue']:.4f})")
    logger.info(f"Best Constraint: {best_constraint['policy']} ({best_constraint['constraint_violation_rate']:.2%})")
    logger.info(f"Fastest:         {fastest['policy']} ({fastest['time_ms']:.1f}ms)")


if __name__ == "__main__":
    main()
