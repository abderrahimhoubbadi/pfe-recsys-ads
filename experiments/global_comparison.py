"""
Global Comparison: All Estimators × All MOO Policies.

Runs all 70 combinations (7 estimators × 10 MOO policies) and generates:
- Performance heatmaps (CTR, Revenue)
- Pareto front per estimator
- Execution time matrix
- Summary tables

Usage:
    python experiments/global_comparison.py
    python experiments/global_comparison.py --n 500   # Fewer iterations
"""
import sys
import os
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import (
    N_ITERATIONS, DIMENSION, N_ARMS, ALPHA,
    EPSILON_THRESHOLD, HIDDEN_DIM, LEARNING_RATE,
    N_ENSEMBLE, TS_VARIANCE, NEURAL_TS_SIGMA,
    PESSIMISM_DECAY, DELAY_WINDOW
)
from src.env.context_generator import ContextGenerator
from src.env.reward_simulator import RewardSimulator

# ============================================================
# All Estimators (7)
# ============================================================
from src.agents.multi_obj_agent import MultiObjectiveLinUCBAgent
from src.agents.thompson_sampling_agent import ThompsonSamplingAgent
from src.agents.neural_ucb_agent import NeuralUCBAgent
from src.agents.neural_ts_agent import NeuralTSAgent
from src.agents.deep_bandit_agent import DeepBanditAgent
from src.agents.offline_online_agent import OfflineOnlineAgent
from src.agents.delayed_feedback_agent import DelayedFeedbackAgent

# ============================================================
# All MOO Policies (10)
# ============================================================
from src.policy.moo_policies import (
    linear_scalarization_policy,
    epsilon_constraint_policy,
    pareto_frontier_policy
)
from src.policy.exact_moo import (
    mobb_policy, two_phase_policy, oss_policy,
    modp_policy, moa_star_policy
)
from src.policy.metaheuristics import nsga2_policy, moead_policy

os.makedirs('metrics', exist_ok=True)


def create_agent(agent_name: str, n_arms: int):
    """Factory function to create agents."""
    agents = {
        'LinUCB': lambda: MultiObjectiveLinUCBAgent(
            n_arms=n_arms, dimension=DIMENSION, alpha=ALPHA
        ),
        'Thompson': lambda: ThompsonSamplingAgent(
            n_arms=n_arms, dimension=DIMENSION, v=TS_VARIANCE
        ),
        'NeuralUCB': lambda: NeuralUCBAgent(
            n_arms=n_arms, dimension=DIMENSION, alpha=ALPHA,
            hidden_dim=HIDDEN_DIM, lr=LEARNING_RATE
        ),
        'NeuralTS': lambda: NeuralTSAgent(
            n_arms=n_arms, dimension=DIMENSION, sigma=NEURAL_TS_SIGMA,
            hidden_dim=HIDDEN_DIM, lr=LEARNING_RATE
        ),
        'DeepBandit': lambda: DeepBanditAgent(
            n_arms=n_arms, dimension=DIMENSION, alpha=ALPHA,
            n_ensemble=N_ENSEMBLE, hidden_dim=HIDDEN_DIM, lr=LEARNING_RATE
        ),
        'Offline2On': lambda: OfflineOnlineAgent(
            n_arms=n_arms, dimension=DIMENSION, alpha=ALPHA,
            pessimism_decay=PESSIMISM_DECAY
        ),
        'DelayedFB': lambda: DelayedFeedbackAgent(
            n_arms=n_arms, dimension=DIMENSION, alpha=ALPHA,
            delay_window=DELAY_WINDOW
        ),
    }
    return agents[agent_name]()


def get_policies():
    """Return all 10 MOO policies."""
    return {
        'Scalar': linear_scalarization_policy({'click': 0.5, 'revenue': 0.5}),
        'ε-Constr': epsilon_constraint_policy('click', 'revenue', epsilon=EPSILON_THRESHOLD),
        'Pareto-Ch': pareto_frontier_policy(),
        'MOBB': mobb_policy(),
        'TwoPhase': two_phase_policy(),
        'OSS': oss_policy(),
        'MODP': modp_policy(),
        'MOA*': moa_star_policy(),
        'NSGA-II': nsga2_policy(),
        'MOEA/D': moead_policy(),
    }


def run_combination(agent, policy_fn, n_iter: int, n_arms: int) -> dict:
    """Run a single (agent, policy) combination."""
    ctx_gen = ContextGenerator(dimension=DIMENSION)
    env = RewardSimulator(dimension=DIMENSION, n_arms=n_arms)

    clicks, revenue, violations = 0, 0.0, 0

    start = time.time()
    for _ in range(n_iter):
        ctx = ctx_gen.get_context()
        arm = agent.select_arm(ctx, policy=policy_fn)
        rewards = env.get_reward(ctx, arm)
        agent.update(ctx, arm, rewards)

        clicks += rewards['click']
        revenue += rewards['revenue']
        if env.get_expected_reward(ctx, arm)['revenue'] < EPSILON_THRESHOLD:
            violations += 1

    duration = time.time() - start

    return {
        'ctr': clicks / n_iter,
        'revenue': revenue / n_iter,
        'violations': violations / n_iter,
        'time_ms': duration * 1000
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=N_ITERATIONS, help='Iterations per combo')
    parser.add_argument('--k', type=int, default=N_ARMS, help='Number of arms (ads)')
    args = parser.parse_args()
    n_iter = args.n
    n_arms = args.k

    agent_names = ['LinUCB', 'Thompson', 'NeuralUCB', 'NeuralTS',
                    'DeepBandit', 'Offline2On', 'DelayedFB']
    policies = get_policies()
    policy_names = list(policies.keys())

    total = len(agent_names) * len(policy_names)
    print(f"{'='*70}")
    print(f"Global Comparison: {len(agent_names)} estimators × {len(policy_names)} policies = {total} combos")
    print(f"K={n_arms} arms, {n_iter} iterations per combo")
    print(f"{'='*70}")

    # Results matrices
    ctr_matrix = np.zeros((len(agent_names), len(policy_names)))
    rev_matrix = np.zeros((len(agent_names), len(policy_names)))
    viol_matrix = np.zeros((len(agent_names), len(policy_names)))
    time_matrix = np.zeros((len(agent_names), len(policy_names)))

    count = 0
    for i, ag_name in enumerate(agent_names):
        for j, pol_name in enumerate(policy_names):
            count += 1
            print(f"[{count}/{total}] {ag_name} × {pol_name}...", end=' ', flush=True)

            try:
                agent = create_agent(ag_name, n_arms)
                result = run_combination(agent, policies[pol_name], n_iter, n_arms)

                ctr_matrix[i, j] = result['ctr']
                rev_matrix[i, j] = result['revenue']
                viol_matrix[i, j] = result['violations']
                time_matrix[i, j] = result['time_ms']

                print(f"CTR={result['ctr']:.3f} Rev={result['revenue']:.3f} "
                      f"T={result['time_ms']:.0f}ms")
            except Exception as e:
                print(f"ERROR: {e}")
                ctr_matrix[i, j] = np.nan
                rev_matrix[i, j] = np.nan

    # ============================================================
    # PLOT 1: CTR Heatmap
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(ctr_matrix, cmap='YlGn', aspect='auto')
    ax.set_xticks(range(len(policy_names)))
    ax.set_yticks(range(len(agent_names)))
    ax.set_xticklabels(policy_names, rotation=45, ha='right')
    ax.set_yticklabels(agent_names)
    ax.set_title('CTR: Estimator × MOO Policy', fontsize=14)
    for i in range(len(agent_names)):
        for j in range(len(policy_names)):
            val = ctr_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, label='CTR')
    plt.tight_layout()
    plt.savefig('metrics/heatmap_ctr.png', dpi=150)
    plt.close()

    # ============================================================
    # PLOT 2: Revenue Heatmap
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(rev_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(policy_names)))
    ax.set_yticks(range(len(agent_names)))
    ax.set_xticklabels(policy_names, rotation=45, ha='right')
    ax.set_yticklabels(agent_names)
    ax.set_title('Revenue: Estimator × MOO Policy', fontsize=14)
    for i in range(len(agent_names)):
        for j in range(len(policy_names)):
            val = rev_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, label='Revenue')
    plt.tight_layout()
    plt.savefig('metrics/heatmap_revenue.png', dpi=150)
    plt.close()

    # ============================================================
    # PLOT 3: Pareto Front per Estimator
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    for i, ag_name in enumerate(agent_names):
        valid = ~np.isnan(ctr_matrix[i])
        ax.scatter(ctr_matrix[i][valid], rev_matrix[i][valid],
                  c=colors[i], marker=markers[i], s=100, label=ag_name, alpha=0.7)
    ax.set_xlabel('CTR', fontsize=12)
    ax.set_ylabel('Revenue', fontsize=12)
    ax.set_title('Pareto Front: All Estimators (each dot = one policy)', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('metrics/pareto_all_estimators.png', dpi=150)
    plt.close()

    # ============================================================
    # PLOT 4: Time Heatmap
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(time_matrix, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(policy_names)))
    ax.set_yticks(range(len(agent_names)))
    ax.set_xticklabels(policy_names, rotation=45, ha='right')
    ax.set_yticklabels(agent_names)
    ax.set_title('Execution Time (ms): Estimator × MOO Policy', fontsize=14)
    for i in range(len(agent_names)):
        for j in range(len(policy_names)):
            val = time_matrix[i, j]
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, label='Time (ms)')
    plt.tight_layout()
    plt.savefig('metrics/heatmap_time.png', dpi=150)
    plt.close()

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("GLOBAL RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Estimator':<12} {'Best Policy (CTR)':<18} {'CTR':>6} {'Best Policy (Rev)':<18} {'Rev':>6}")
    print("-" * 70)
    for i, ag in enumerate(agent_names):
        valid = ~np.isnan(ctr_matrix[i])
        if np.any(valid):
            best_ctr_j = np.nanargmax(ctr_matrix[i])
            best_rev_j = np.nanargmax(rev_matrix[i])
            print(f"{ag:<12} {policy_names[best_ctr_j]:<18} {ctr_matrix[i, best_ctr_j]:>6.3f} "
                  f"{policy_names[best_rev_j]:<18} {rev_matrix[i, best_rev_j]:>6.3f}")

    # Overall best
    best_idx = np.unravel_index(np.nanargmax(ctr_matrix), ctr_matrix.shape)
    print(f"\n🏆 Best CTR: {agent_names[best_idx[0]]} × {policy_names[best_idx[1]]} = {ctr_matrix[best_idx]:.4f}")
    best_idx = np.unravel_index(np.nanargmax(rev_matrix), rev_matrix.shape)
    print(f"🏆 Best Revenue: {agent_names[best_idx[0]]} × {policy_names[best_idx[1]]} = {rev_matrix[best_idx]:.4f}")

    print(f"\n✅ Plots saved to metrics/")
    print("  - heatmap_ctr.png")
    print("  - heatmap_revenue.png")
    print("  - pareto_all_estimators.png")
    print("  - heatmap_time.png")


if __name__ == "__main__":
    main()
