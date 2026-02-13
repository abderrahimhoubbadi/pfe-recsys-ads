"""
Exact MOO Comparison with Visual Plots.

Compares all 10 policies and generates:
- Pareto front scatter plot
- Bar chart comparison
- Execution time comparison
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import (
    N_ITERATIONS, DIMENSION, N_ARMS, ALPHA, 
    EPSILON_THRESHOLD
)
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
    moead_policy
)

# Ensure metrics directory exists
os.makedirs('metrics', exist_ok=True)


def run_experiment(policy_fn, n_iter: int) -> dict:
    """Run experiment and return detailed metrics."""
    ctx_gen = ContextGenerator(dimension=DIMENSION)
    env = RewardSimulator(dimension=DIMENSION, n_arms=N_ARMS)
    agent = MultiObjectiveLinUCBAgent(
        n_arms=N_ARMS,
        dimension=DIMENSION,
        alpha=ALPHA,
        objectives=['click', 'revenue']
    )
    
    clicks, revenues = 0, 0.0
    violations = 0
    
    start = time.time()
    for _ in range(n_iter):
        ctx = ctx_gen.get_context()
        arm = agent.select_arm(ctx, policy=policy_fn)
        rewards = env.get_reward(ctx, arm)
        agent.update(ctx, arm, rewards)
        
        clicks += rewards['click']
        revenues += rewards['revenue']
        if env.get_expected_reward(ctx, arm)['revenue'] < EPSILON_THRESHOLD:
            violations += 1
    
    duration = time.time() - start
    
    return {
        'ctr': clicks / n_iter,
        'revenue': revenues / n_iter,
        'violations': violations / n_iter,
        'time_ms': duration * 1000
    }


def main():
    print("=" * 60)
    print("MOO Policy Comparison with Visual Analysis")
    print("=" * 60)
    
    # Define all policies with categories
    policies = {
        # Scalarization
        'Scalar-0.5': (linear_scalarization_policy({'click': 0.5, 'revenue': 0.5}), 'Scalarization', 'blue'),
        'ε-Constraint': (epsilon_constraint_policy('click', 'revenue', epsilon=EPSILON_THRESHOLD), 'Scalarization', 'blue'),
        'Pareto-Cheb': (pareto_frontier_policy(), 'Scalarization', 'blue'),
        
        # Exact
        'MOBB': (mobb_policy(), 'Exact', 'green'),
        'Two-Phase': (two_phase_policy(), 'Exact', 'green'),
        'OSS': (oss_policy(), 'Exact', 'green'),
        'MODP': (modp_policy(), 'Exact', 'green'),
        'MOA*': (moa_star_policy(), 'Exact', 'green'),
        
        # Metaheuristics
        'NSGA-II': (nsga2_policy(), 'Metaheuristic', 'red'),
        'MOEA/D': (moead_policy(), 'Metaheuristic', 'red'),
    }
    
    # Run experiments
    results = {}
    for name, (policy_fn, category, color) in policies.items():
        print(f"Running {name}...")
        metrics = run_experiment(policy_fn, N_ITERATIONS)
        results[name] = {**metrics, 'category': category, 'color': color}
    
    # === PLOT 1: Pareto Front (CTR vs Revenue) ===
    plt.figure(figsize=(12, 8))
    
    for name, data in results.items():
        marker = 'o' if data['category'] == 'Scalarization' else ('s' if data['category'] == 'Exact' else '^')
        plt.scatter(data['ctr'], data['revenue'], 
                   c=data['color'], marker=marker, s=150, label=name, alpha=0.8)
        plt.annotate(name, (data['ctr'], data['revenue']), 
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    plt.xlabel('CTR (Click-Through Rate)', fontsize=12)
    plt.ylabel('Average Revenue', fontsize=12)
    plt.title('Pareto Front: CTR vs Revenue (All 10 Policies)', fontsize=14)
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('metrics/pareto_front.png', dpi=150)
    plt.close()
    print("Saved: metrics/pareto_front.png")
    
    # === PLOT 2: Bar Chart Comparison ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = list(results.keys())
    colors = [results[n]['color'] for n in names]
    
    # CTR
    ax = axes[0, 0]
    ax.bar(names, [results[n]['ctr'] for n in names], color=colors, alpha=0.7)
    ax.set_ylabel('CTR')
    ax.set_title('Click-Through Rate')
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Revenue
    ax = axes[0, 1]
    ax.bar(names, [results[n]['revenue'] for n in names], color=colors, alpha=0.7)
    ax.set_ylabel('Revenue')
    ax.set_title('Average Revenue')
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Violations
    ax = axes[1, 0]
    ax.bar(names, [results[n]['violations'] * 100 for n in names], color=colors, alpha=0.7)
    ax.set_ylabel('Violation Rate (%)')
    ax.set_title('Constraint Violations')
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Time
    ax = axes[1, 1]
    ax.bar(names, [results[n]['time_ms'] for n in names], color=colors, alpha=0.7)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Execution Time')
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('metrics/policy_comparison.png', dpi=150)
    plt.close()
    print("Saved: metrics/policy_comparison.png")
    
    # === PLOT 3: Category Comparison ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Scalarization', 'Exact', 'Metaheuristic']
    cat_colors = ['blue', 'green', 'red']
    
    for i, (cat, color) in enumerate(zip(categories, cat_colors)):
        cat_results = [v for k, v in results.items() if v['category'] == cat]
        if cat_results:
            avg_ctr = np.mean([r['ctr'] for r in cat_results])
            avg_rev = np.mean([r['revenue'] for r in cat_results])
            ax.scatter(avg_ctr, avg_rev, c=color, s=300, marker='D', 
                      label=f'{cat} (avg)', edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Average CTR', fontsize=12)
    ax.set_ylabel('Average Revenue', fontsize=12)
    ax.set_title('Category Comparison: Scalarization vs Exact vs Metaheuristic', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('metrics/category_comparison.png', dpi=150)
    plt.close()
    print("Saved: metrics/category_comparison.png")
    
    # === Summary Table ===
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Policy':<15} {'Category':<15} {'CTR':>8} {'Revenue':>10} {'Violations':>12} {'Time(ms)':>10}")
    print("-" * 80)
    for name, data in results.items():
        print(f"{name:<15} {data['category']:<15} {data['ctr']:>8.4f} {data['revenue']:>10.4f} "
              f"{data['violations']:>11.2%} {data['time_ms']:>10.1f}")
    
    print("\n✅ All plots saved to metrics/ folder")


if __name__ == "__main__":
    main()
