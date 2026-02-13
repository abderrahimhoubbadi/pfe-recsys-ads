"""
MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) Policy.

Decomposes the multi-objective problem into scalar subproblems.
"""
import numpy as np
from typing import List, Tuple, Callable

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def moead_policy(n_weights: int = 5, method: str = 'tchebycheff') -> Callable:
    """
    MOEA/D inspired policy for arm selection.
    
    Decomposes the problem using weight vectors and solves scalar subproblems.
    
    Args:
        n_weights: Number of weight vectors to use
        method: Aggregation method ('tchebycheff' or 'weighted_sum')
    
    Returns:
        Policy function
    """
    def policy(predictions: List[dict]) -> int:
        """
        Select arm using MOEA/D approach.
        """
        n_arms = len(predictions)
        
        # Generate uniformly distributed weight vectors
        weights = []
        for i in range(n_weights):
            w1 = i / (n_weights - 1) if n_weights > 1 else 0.5
            w2 = 1 - w1
            weights.append(np.array([w1, w2]))
        
        # Extract solutions
        solutions = []
        for arm_idx, pred in enumerate(predictions):
            ucb_click = pred['click'][1]
            ucb_revenue = pred['revenue'][1]
            solutions.append((arm_idx, np.array([ucb_click, ucb_revenue])))
        
        # Compute reference point (ideal)
        all_vecs = [vec for _, vec in solutions]
        z_star = np.max(all_vecs, axis=0)  # Ideal point
        
        # For each weight vector, find the best arm
        subproblem_solutions = []
        
        for w in weights:
            if method == 'tchebycheff':
                # Tchebycheff: minimize max(w_i * |z*_i - f_i|)
                # For maximization: minimize max(w_i * (z*_i - f_i))
                best_arm = None
                best_score = float('inf')
                
                for arm, vec in solutions:
                    score = np.max(w * (z_star - vec + 1e-6))
                    if score < best_score:
                        best_score = score
                        best_arm = arm
                
                if best_arm is not None:
                    subproblem_solutions.append(best_arm)
            
            else:  # weighted_sum
                best_arm = max(solutions, key=lambda x: np.dot(w, x[1]))[0]
                subproblem_solutions.append(best_arm)
        
        # Aggregate: most frequently selected arm
        if not subproblem_solutions:
            return 0
        
        arm_counts = {}
        for arm in subproblem_solutions:
            arm_counts[arm] = arm_counts.get(arm, 0) + 1
        
        # Return most frequently selected (consensus arm)
        return max(arm_counts.keys(), key=lambda a: arm_counts[a])
    
    return policy


def moead_policy_adaptive() -> Callable:
    """
    Adaptive MOEA/D that adjusts weights based on historical performance.
    """
    weight_performance = {}  # Track which weights work best
    
    def policy(predictions: List[dict]) -> int:
        """
        Select arm using adaptive MOEA/D.
        """
        n_arms = len(predictions)
        
        # Fixed set of weight vectors
        weights = [
            np.array([1.0, 0.0]),   # Pure click
            np.array([0.75, 0.25]),
            np.array([0.5, 0.5]),   # Balanced
            np.array([0.25, 0.75]),
            np.array([0.0, 1.0]),   # Pure revenue
        ]
        
        solutions = []
        for arm_idx, pred in enumerate(predictions):
            ucb_click = pred['click'][1]
            ucb_revenue = pred['revenue'][1]
            solutions.append((arm_idx, np.array([ucb_click, ucb_revenue])))
        
        # Compute ideal point
        all_vecs = [vec for _, vec in solutions]
        z_star = np.max(all_vecs, axis=0)
        
        # Solve each subproblem and track
        arm_scores = {arm: 0.0 for arm in range(n_arms)}
        
        for i, w in enumerate(weights):
            # Tchebycheff approach
            best_arm = None
            best_score = float('inf')
            
            for arm, vec in solutions:
                score = np.max(w * (z_star - vec + 1e-6))
                if score < best_score:
                    best_score = score
                    best_arm = arm
            
            if best_arm is not None:
                # Weight contribution by past performance
                weight_key = tuple(w.tolist())
                perf_factor = weight_performance.get(weight_key, 1.0)
                arm_scores[best_arm] += perf_factor
        
        return max(arm_scores.keys(), key=lambda a: arm_scores[a])
    
    return policy
