"""
Multi-Objective A* (MOA*) Policy.

Uses heuristic-guided search in objective space with bounds.
"""
import numpy as np
from typing import List, Tuple, Callable, Set
import heapq

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.policy.pareto_utils import get_pareto_front, is_dominated


def moa_star_policy() -> Callable:
    """
    Multi-Objective A* policy.
    
    Uses priority queue with heuristic to efficiently explore arm space.
    
    Returns:
        Policy function
    """
    def policy(predictions: List[dict]) -> int:
        """
        Select arm using MOA*.
        """
        n_arms = len(predictions)
        
        # Extract predictions with bounds
        arms_data = []
        for arm_idx, pred in enumerate(predictions):
            mean_click = pred['click'][0]
            ucb_click = pred['click'][1]
            mean_revenue = pred['revenue'][0]
            ucb_revenue = pred['revenue'][1]
            
            # Lower bound (mean), Upper bound (UCB)
            lower = np.array([mean_click, mean_revenue])
            upper = np.array([ucb_click, ucb_revenue])
            
            # Heuristic: potential improvement (gap between lower and upper)
            heuristic = np.sum(upper - lower)
            
            arms_data.append((arm_idx, lower, upper, heuristic))
        
        # Priority queue: (negative_heuristic, arm_idx, upper_bound)
        # Negative because heapq is min-heap
        open_set = []
        for arm_idx, lower, upper, h in arms_data:
            # Priority: combination of upper bound quality and heuristic
            priority = -np.sum(upper)  # Higher UCB = lower priority value
            heapq.heappush(open_set, (priority, arm_idx, upper.tolist()))
        
        # Track best solutions found
        pareto_solutions = []
        explored = set()
        
        while open_set:
            _, arm_idx, upper_list = heapq.heappop(open_set)
            
            if arm_idx in explored:
                continue
            explored.add(arm_idx)
            
            upper = np.array(upper_list)
            
            # Check if this arm can contribute to Pareto front
            is_dominated_flag = False
            for _, existing_vec in pareto_solutions:
                if is_dominated(upper, existing_vec, maximize=True):
                    is_dominated_flag = True
                    break
            
            if not is_dominated_flag:
                # Add to Pareto solutions and remove dominated ones
                new_pareto = [(a, v) for a, v in pareto_solutions 
                             if not is_dominated(v, upper, maximize=True)]
                new_pareto.append((arm_idx, upper))
                pareto_solutions = new_pareto
            
            # Early termination: if we've explored enough
            if len(explored) >= min(n_arms, 10):
                break
        
        # Get final Pareto front
        pareto_front = get_pareto_front(pareto_solutions, maximize=True)
        
        if not pareto_front:
            return 0
        
        if len(pareto_front) == 1:
            return pareto_front[0][0]
        
        # Tie-breaking: prefer arm with highest minimum objective
        # (maximin approach for robustness)
        return max(pareto_front, key=lambda x: np.min(x[1]))[0]
    
    return policy
