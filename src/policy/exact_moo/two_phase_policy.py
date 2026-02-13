"""
Two-Phase Method Policy (Ulungu & Teghem).

Phase 1: Find supported solutions (on the convex hull)
Phase 2: Find non-supported solutions (inside the hull)
"""
import numpy as np
from typing import List, Tuple, Callable

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.policy.pareto_utils import get_pareto_front, is_dominated


def two_phase_policy() -> Callable:
    """
    Two-Phase method for bi-objective optimization.
    
    Phase 1: Scalarization to find supported solutions (convex hull vertices)
    Phase 2: ε-constraint to find non-supported solutions
    
    Returns:
        Policy function
    """
    def policy(predictions: List[dict]) -> int:
        """
        Select arm using Two-Phase method.
        """
        n_arms = len(predictions)
        
        # Extract UCB values
        solutions = []
        for arm_idx, pred in enumerate(predictions):
            ucb_click = pred['click'][1]
            ucb_revenue = pred['revenue'][1]
            solutions.append((arm_idx, np.array([ucb_click, ucb_revenue])))
        
        # PHASE 1: Find supported solutions using weighted sum
        supported = []
        
        # Extreme points
        # Max click
        arm_max_click = max(solutions, key=lambda x: x[1][0])
        supported.append(arm_max_click)
        
        # Max revenue
        arm_max_revenue = max(solutions, key=lambda x: x[1][1])
        if arm_max_revenue[0] != arm_max_click[0]:
            supported.append(arm_max_revenue)
        
        # Find intermediate supported solutions
        # Try multiple weight combinations
        for w in np.linspace(0.1, 0.9, 9):
            weights = np.array([w, 1 - w])
            scores = [(arm, np.dot(weights, vec)) for arm, vec in solutions]
            best = max(scores, key=lambda x: x[1])
            best_arm = best[0]
            
            # Check if already in supported
            if not any(s[0] == best_arm for s in supported):
                supported.append(solutions[best_arm])
        
        # PHASE 2: Find non-supported solutions between adjacent pairs
        # Sort supported by first objective
        supported = sorted(supported, key=lambda x: x[1][0])
        
        all_pareto = list(supported)
        
        for i in range(len(supported) - 1):
            arm_i, vec_i = supported[i]
            arm_j, vec_j = supported[i + 1]
            
            # ε-constraint: maximize click s.t. revenue > vec_i[1]
            epsilon = vec_i[1]
            
            for arm, vec in solutions:
                if arm != arm_i and arm != arm_j:
                    # Check if in the region between i and j
                    if vec[1] > epsilon and vec[0] > vec_i[0] and vec[0] < vec_j[0]:
                        # Check if non-dominated
                        if not any(is_dominated(vec, s[1], maximize=True) 
                                  for s in all_pareto):
                            all_pareto.append((arm, vec))
        
        # Get final Pareto front
        pareto_front = get_pareto_front(all_pareto, maximize=True)
        
        if len(pareto_front) == 1:
            return pareto_front[0][0]
        
        # Tie-breaking: balanced score
        return max(pareto_front, key=lambda x: np.sum(x[1]))[0]
    
    return policy
