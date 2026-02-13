"""
Multi-Objective Dynamic Programming (MODP) Policy.

Adapted DP approach that maintains Pareto sets during arm evaluation.
"""
import numpy as np
from typing import List, Tuple, Callable

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.policy.pareto_utils import get_pareto_front, update_pareto_set


def modp_policy() -> Callable:
    """
    Multi-Objective Dynamic Programming policy.
    
    Iteratively builds Pareto set by evaluating arms one by one.
    
    Returns:
        Policy function
    """
    def policy(predictions: List[dict]) -> int:
        """
        Select arm using MODP approach.
        """
        n_arms = len(predictions)
        
        # Initialize empty Pareto set
        pareto_set = []
        
        # Process each arm (DP-style iteration)
        for arm_idx, pred in enumerate(predictions):
            ucb_click = pred['click'][1]
            ucb_revenue = pred['revenue'][1]
            obj_vector = np.array([ucb_click, ucb_revenue])
            
            # Update Pareto set with new solution
            pareto_set = update_pareto_set(pareto_set, (arm_idx, obj_vector), maximize=True)
        
        if not pareto_set:
            return 0
        
        if len(pareto_set) == 1:
            return pareto_set[0][0]
        
        # Tie-breaking: maximize product (geometric mean approach)
        # This prefers balanced solutions
        return max(pareto_set, key=lambda x: np.prod(x[1] + 1e-6))[0]
    
    return policy


def modp_policy_with_memory() -> Callable:
    """
    Enhanced MODP with memory of previous decisions.
    
    Maintains state across calls for improved convergence.
    """
    history = []
    
    def policy(predictions: List[dict]) -> int:
        """
        Select arm using MODP with historical memory.
        """
        nonlocal history
        
        n_arms = len(predictions)
        
        # Build current Pareto set
        pareto_set = []
        
        for arm_idx, pred in enumerate(predictions):
            ucb_click = pred['click'][1]
            ucb_revenue = pred['revenue'][1]
            obj_vector = np.array([ucb_click, ucb_revenue])
            pareto_set = update_pareto_set(pareto_set, (arm_idx, obj_vector), maximize=True)
        
        if not pareto_set:
            return 0
        
        # Track arm selection frequency
        arm_counts = {}
        for arm, _ in history[-100:]:  # Last 100 decisions
            arm_counts[arm] = arm_counts.get(arm, 0) + 1
        
        # Prefer less-selected Pareto-optimal arms (exploration)
        if arm_counts:
            selected = min(pareto_set, 
                          key=lambda x: arm_counts.get(x[0], 0))
        else:
            selected = max(pareto_set, key=lambda x: np.sum(x[1]))
        
        history.append(selected)
        return selected[0]
    
    return policy
