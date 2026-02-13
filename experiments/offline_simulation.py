"""
Offline Simulation with Multi-Objective LinUCB.

Purpose: Debug and verify that the MOO agent learns both objectives correctly.
"""
import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import N_ITERATIONS, N_ARMS, DIMENSION, ALPHA, OBJECTIVES
from src.env.context_generator import ContextGenerator
from src.env.reward_simulator import RewardSimulator
from src.agents.multi_obj_agent import MultiObjectiveLinUCBAgent
from src.policy.moo_policies import epsilon_constraint_policy


def run_simulation(n_iterations=None, n_arms=None, dimension=None, alpha=None):
    # Use config defaults if not specified
    n_iterations = n_iterations or N_ITERATIONS
    n_arms = n_arms or N_ARMS
    dimension = dimension or DIMENSION
    alpha = alpha or ALPHA
    
    print(f"--- Démarrage Simulation Offline (Multi-Objectif) ---")
    print(f"Iterations: {n_iterations}, Arms: {n_arms}, Dim: {dimension}, Alpha: {alpha}")
    
    # 1. Init Components
    ctx_gen = ContextGenerator(dimension=dimension)
    env = RewardSimulator(dimension=dimension, n_arms=n_arms)
    agent = MultiObjectiveLinUCBAgent(
        n_arms=n_arms, 
        dimension=dimension, 
        alpha=alpha,
        objectives=OBJECTIVES
    )
    
    # Use epsilon-constraint policy by default
    policy = epsilon_constraint_policy('click', 'revenue', epsilon=0.3)
    
    # Metrics per objective
    cumulative_clicks = 0
    cumulative_revenue = 0.0
    cumulative_regret_click = 0.0
    cumulative_regret_revenue = 0.0
    
    start_time = time.time()
    
    for t in range(n_iterations):
        # 1. Get Context
        context = ctx_gen.get_context()
        
        # 2. Agent Choose Arm using MOO policy
        chosen_arm = agent.select_arm(context, policy=policy)
        
        # 3. Get Rewards (Dict: {'click': int, 'revenue': float})
        rewards = env.get_reward(context, chosen_arm)
        
        # 4. Update Agent with all objectives
        agent.update(context, chosen_arm, rewards)
        
        # --- Evaluation (Oracle) ---
        optimal_click = env.get_optimal_arm(context, objective='click')
        optimal_revenue = env.get_optimal_arm(context, objective='revenue')
        
        expected_chosen = env.get_expected_reward(context, chosen_arm)
        expected_opt_click = env.get_expected_reward(context, optimal_click)
        expected_opt_revenue = env.get_expected_reward(context, optimal_revenue)
        
        regret_click = expected_opt_click['click'] - expected_chosen['click']
        regret_revenue = expected_opt_revenue['revenue'] - expected_chosen['revenue']
        
        cumulative_clicks += rewards['click']
        cumulative_revenue += rewards['revenue']
        cumulative_regret_click += regret_click
        cumulative_regret_revenue += regret_revenue
        
        if (t + 1) % 500 == 0:
            ctr = cumulative_clicks / (t + 1)
            avg_rev = cumulative_revenue / (t + 1)
            avg_reg_c = cumulative_regret_click / (t + 1)
            avg_reg_r = cumulative_regret_revenue / (t + 1)
            print(f"Step {t+1}/{n_iterations} | CTR: {ctr:.4f} | Avg Revenue: {avg_rev:.4f} | "
                  f"Regret(C): {avg_reg_c:.4f} | Regret(R): {avg_reg_r:.4f}")

    duration = time.time() - start_time
    print(f"\n--- Simulation Terminée en {duration:.2f}s ---")
    print(f"Final CTR: {cumulative_clicks / n_iterations:.4f}")
    print(f"Final Avg Revenue: {cumulative_revenue / n_iterations:.4f}")
    
    return {'clicks': cumulative_clicks, 'revenue': cumulative_revenue}


if __name__ == "__main__":
    run_simulation()
