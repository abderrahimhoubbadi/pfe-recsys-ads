"""
Objective Space Search (OSS) Policy - Aneja-Nair Algorithm.

Recursively divides the objective space to find all Pareto-optimal solutions.
"""
import numpy as np
from typing import List, Tuple, Callable, Optional

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.policy.pareto_utils import get_pareto_front


def oss_policy() -> Callable:
    """
    Objective Space Search (Aneja-Nair) policy.
    
    Divides objective space recursively to find complete Pareto front.
    
    Returns:
        Policy function
    """
    def policy(predictions: List[dict]) -> int:
        """
        Select arm using OSS.
        """
        n_arms = len(predictions)
        
        # Extract UCB values
        solutions = []
        for arm_idx, pred in enumerate(predictions):
            ucb_click = pred['click'][1]
            ucb_revenue = pred['revenue'][1]
            solutions.append((arm_idx, np.array([ucb_click, ucb_revenue])))
        
        def find_optimal_in_region(sol_list: List[Tuple[int, np.ndarray]], 
                                   objective_idx: int) -> Optional[Tuple[int, np.ndarray]]:
            """Find optimal solution for a specific objective."""
            if not sol_list:
                return None
            return max(sol_list, key=lambda x: x[1][objective_idx])
        
        def oss_recursive(sol_list: List[Tuple[int, np.ndarray]], 
                          z1: Optional[Tuple[int, np.ndarray]] = None,
                          z2: Optional[Tuple[int, np.ndarray]] = None,
                          pareto_set: List = None) -> List[Tuple[int, np.ndarray]]:
            """
            Recursive OSS algorithm.
            
            Args:
                sol_list: Current solution candidates
                z1: Upper left corner of search region
                z2: Lower right corner of search region
                pareto_set: Accumulated Pareto solutions
            """
            if pareto_set is None:
                pareto_set = []
            
            if len(sol_list) <= 1:
                if sol_list:
                    pareto_set.append(sol_list[0])
                return pareto_set
            
            # Find extremes if not provided
            if z1 is None:
                z1 = find_optimal_in_region(sol_list, 1)  # Max revenue
            if z2 is None:
                z2 = find_optimal_in_region(sol_list, 0)  # Max click
            
            if z1 is None or z2 is None:
                return pareto_set
            
            # If same solution, add and return
            if z1[0] == z2[0]:
                if not any(s[0] == z1[0] for s in pareto_set):
                    pareto_set.append(z1)
                return pareto_set
            
            # Add extremes to Pareto set
            if not any(s[0] == z1[0] for s in pareto_set):
                pareto_set.append(z1)
            if not any(s[0] == z2[0] for s in pareto_set):
                pareto_set.append(z2)
            
            # Find solution in the middle region
            # Look for arm that maximizes weighted sum
            mid_weight = 0.5
            weights = np.array([mid_weight, 1 - mid_weight])
            
            # Filter solutions in the region between z1 and z2
            region_sols = [
                (arm, vec) for arm, vec in sol_list
                if vec[0] >= z1[1][0] and vec[0] <= z2[1][0]
                and vec[1] >= z2[1][1] and vec[1] <= z1[1][1]
            ]
            
            if region_sols:
                mid_sol = max(region_sols, key=lambda x: np.dot(weights, x[1]))
                
                if mid_sol[0] != z1[0] and mid_sol[0] != z2[0]:
                    if not any(s[0] == mid_sol[0] for s in pareto_set):
                        pareto_set.append(mid_sol)
                    
                    # Recurse on left and right regions
                    left_sols = [(a, v) for a, v in sol_list 
                                if v[0] < mid_sol[1][0]]
                    right_sols = [(a, v) for a, v in sol_list 
                                 if v[0] > mid_sol[1][0]]
                    
                    if left_sols:
                        oss_recursive(left_sols, z1, mid_sol, pareto_set)
                    if right_sols:
                        oss_recursive(right_sols, mid_sol, z2, pareto_set)
            
            return pareto_set
        
        # Run OSS
        pareto_solutions = oss_recursive(solutions)
        
        # Get clean Pareto front
        pareto_front = get_pareto_front(pareto_solutions, maximize=True)
        
        if not pareto_front:
            # Fallback
            return max(range(n_arms), key=lambda i: 
                      predictions[i]['click'][1] + predictions[i]['revenue'][1])
        
        if len(pareto_front) == 1:
            return pareto_front[0][0]
        
        # Tie-breaking: balanced score
        return max(pareto_front, key=lambda x: np.sum(x[1]))[0]
    
    return policy
