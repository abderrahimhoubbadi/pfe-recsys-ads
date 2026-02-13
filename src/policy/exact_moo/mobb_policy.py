"""
Multi-Objective Branch and Bound (MOBB) Policy.

An exact method that systematically explores all arms and prunes
those dominated by known solutions.
"""
import numpy as np
from typing import List, Tuple, Callable

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.policy.pareto_utils import get_pareto_front, is_dominated


def mobb_policy() -> Callable:
    """
    Multi-Objective Branch and Bound policy.
    
    Enumerates all arms, computes UCB for each objective,
    and returns the Pareto-optimal arm with best combined score.
    
    Returns:
        Policy function that takes predictions and returns arm index
    """
    def policy(predictions: List[dict]) -> int:
        """
        Select arm using MOBB.
        
        Args:
            predictions: List of dicts, each containing objective predictions
                        Format: [{'click': (mean, ucb), 'revenue': (mean, ucb)}, ...]
        
        Returns:
            Index of selected arm
        """
        # Extract UCB values for all arms and objectives
        solutions = []
        for arm_idx, pred in enumerate(predictions):
            # Use UCB values (optimistic estimates)
            ucb_vector = np.array([pred['click'][1], pred['revenue'][1]])
            solutions.append((arm_idx, ucb_vector))
        
        # Get Pareto front
        pareto_front = get_pareto_front(solutions, maximize=True)
        
        if len(pareto_front) == 1:
            return pareto_front[0][0]
        
        # If multiple Pareto-optimal solutions, use tie-breaking
        # Option 1: Sum of objectives (balanced)
        best_arm = max(pareto_front, key=lambda x: np.sum(x[1]))[0]
        
        return best_arm
    
    return policy


def mobb_policy_with_bounds() -> Callable:
    """
    Enhanced MOBB with explicit bound computation and pruning.
    
    Uses lower bounds (mean) and upper bounds (UCB) for pruning.
    """
    def policy(predictions: List[dict]) -> int:
        """
        Select arm using MOBB with bounds.
        """
        n_arms = len(predictions)
        
        # Compute bounds for each arm
        bounds = []
        for arm_idx, pred in enumerate(predictions):
            lower = np.array([pred['click'][0], pred['revenue'][0]])  # mean
            upper = np.array([pred['click'][1], pred['revenue'][1]])  # UCB
            bounds.append((arm_idx, lower, upper))
        
        # Initialize with all arms as candidates
        candidates = list(range(n_arms))
        best_solutions = []  # Known Pareto-optimal solutions
        
        # Branch and Bound
        pruned = set()
        
        for arm_idx in candidates:
            if arm_idx in pruned:
                continue
            
            _, lower_i, upper_i = bounds[arm_idx]
            
            # Check if this arm can be pruned
            can_prune = False
            for other_idx in candidates:
                if other_idx == arm_idx or other_idx in pruned:
                    continue
                    
                _, lower_j, _ = bounds[other_idx]
                
                # Prune if other arm's lower bound dominates this arm's upper bound
                if is_dominated(upper_i, lower_j, maximize=True):
                    can_prune = True
                    break
            
            if not can_prune:
                best_solutions.append((arm_idx, upper_i))
            else:
                pruned.add(arm_idx)
        
        # Get final Pareto front from non-pruned solutions
        pareto_front = get_pareto_front(best_solutions, maximize=True)
        
        if not pareto_front:
            # Fallback: return arm with best mean
            return max(range(n_arms), 
                      key=lambda i: sum(predictions[i][obj][0] 
                                       for obj in predictions[i]))
        
        # Tie-breaking: balanced score
        return max(pareto_front, key=lambda x: np.sum(x[1]))[0]
    
    return policy
