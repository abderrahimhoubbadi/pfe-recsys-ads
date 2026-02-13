"""
Benchmark for Large K (Number of Arms).

Tests scalability of all MOO policies with varying K values.

Usage:
    python experiments/benchmark_large_k.py              # Default K values
    python experiments/benchmark_large_k.py --k 50 100   # Custom K values
    python experiments/benchmark_large_k.py --k 200 --n 100  # K=200, 100 iterations
"""
import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env.context_generator import ContextGenerator
from src.env.reward_simulator import RewardSimulator
from src.agents.multi_obj_agent import MultiObjectiveLinUCBAgent

# All policies
from src.policy.moo_policies import (
    linear_scalarization_policy,
    epsilon_constraint_policy,
    pareto_frontier_policy
)
from src.policy.exact_moo import (
    mobb_policy,
    two_phase_policy,
    oss_policy,
    modp_policy,
    moa_star_policy
)
from src.policy.metaheuristics import (
    nsga2_policy,
    nsga2_policy_lite,
    moead_policy
)

# Ensure metrics directory exists
os.makedirs('metrics', exist_ok=True)


def run_single_benchmark(policy_fn, k: int, n_iter: int, dimension: int = 5) -> dict:
    """Run benchmark for a single policy."""
    ctx_gen = ContextGenerator(dimension=dimension)
    env = RewardSimulator(dimension=dimension, n_arms=k)
    agent = MultiObjectiveLinUCBAgent(n_arms=k, dimension=dimension, alpha=0.2)
    
    clicks, revenue = 0, 0.0
    
    start = time.time()
    for _ in range(n_iter):
        ctx = ctx_gen.get_context()
        arm = agent.select_arm(ctx, policy=policy_fn)
        rewards = env.get_reward(ctx, arm)
        agent.update(ctx, arm, rewards)
        clicks += rewards['click']
        revenue += rewards['revenue']
    
    duration = time.time() - start
    
    return {
        'time_ms': duration * 1000,
        'ms_per_iter': duration / n_iter * 1000,
        'ctr': clicks / n_iter,
        'revenue': revenue / n_iter
    }


def get_policies():
    """Return all policies to benchmark."""
    return {
        # Scalarization
        'ε-Constraint': epsilon_constraint_policy('click', 'revenue', epsilon=0.3),
        'Scalarization': linear_scalarization_policy({'click': 0.5, 'revenue': 0.5}),
        'Pareto-Cheb': pareto_frontier_policy(),
        
        # Exact
        'MOBB': mobb_policy(),
        'Two-Phase': two_phase_policy(),
        'OSS': oss_policy(),
        'MODP': modp_policy(),
        'MOA*': moa_star_policy(),
        
        # Metaheuristics
        'NSGA-II': nsga2_policy(),
        'NSGA-II Lite': nsga2_policy_lite(),
        'MOEA/D': moead_policy(),
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark MOO policies with large K')
    parser.add_argument('--k', nargs='+', type=int, default=[5, 20, 50, 100],
                       help='Values of K to test (default: 5 20 50 100)')
    parser.add_argument('--n', type=int, default=300,
                       help='Number of iterations per test (default: 300)')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate scalability plot')
    args = parser.parse_args()
    
    k_values = args.k
    n_iter = args.n
    policies = get_policies()
    
    print("=" * 70)
    print(f"MOO Scalability Benchmark - K values: {k_values}, Iterations: {n_iter}")
    print("=" * 70)
    
    # Store results for plotting
    results = {name: {'k': [], 'time': []} for name in policies}
    
    for k in k_values:
        print(f"\n--- K = {k} arms ---")
        print(f"{'Policy':<18} {'Time (ms)':<12} {'ms/iter':<10} {'CTR':<8} {'Revenue':<8}")
        print("-" * 60)
        
        for name, policy_fn in policies.items():
            try:
                result = run_single_benchmark(policy_fn, k, n_iter)
                results[name]['k'].append(k)
                results[name]['time'].append(result['ms_per_iter'])
                print(f"{name:<18} {result['time_ms']:<12.1f} {result['ms_per_iter']:<10.2f} "
                      f"{result['ctr']:<8.4f} {result['revenue']:<8.4f}")
            except Exception as e:
                print(f"{name:<18} ERROR: {e}")
                results[name]['k'].append(k)
                results[name]['time'].append(np.nan)
    
    # Generate plot
    if args.plot and len(k_values) > 1:
        plt.figure(figsize=(12, 8))
        
        colors = {
            'ε-Constraint': 'blue', 'Scalarization': 'lightblue', 'Pareto-Cheb': 'darkblue',
            'MOBB': 'green', 'Two-Phase': 'lightgreen', 'OSS': 'darkgreen',
            'MODP': 'olive', 'MOA*': 'lime',
            'NSGA-II': 'red', 'NSGA-II Lite': 'orange', 'MOEA/D': 'salmon'
        }
        
        for name, data in results.items():
            if data['k'] and not all(np.isnan(data['time'])):
                plt.plot(data['k'], data['time'], 'o-', label=name, 
                        color=colors.get(name, 'gray'), linewidth=2, markersize=8)
        
        plt.xlabel('Number of Arms (K)', fontsize=12)
        plt.ylabel('Time per Iteration (ms)', fontsize=12)
        plt.title('MOO Policy Scalability: Time vs Number of Arms', fontsize=14)
        plt.legend(loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visibility
        plt.tight_layout()
        plt.savefig('metrics/scalability_k.png', dpi=150)
        plt.close()
        print(f"\n✅ Scalability plot saved to metrics/scalability_k.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Best policies by K")
    print("=" * 70)
    
    for k in k_values:
        k_idx = k_values.index(k)
        times = {name: results[name]['time'][k_idx] 
                for name in policies 
                if k_idx < len(results[name]['time']) and not np.isnan(results[name]['time'][k_idx])}
        if times:
            best = min(times, key=times.get)
            print(f"K={k:>3}: {best} ({times[best]:.2f} ms/iter)")


if __name__ == "__main__":
    main()
