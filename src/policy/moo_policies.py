"""
Multi-Objective Optimization (MOO) Decision Policies.

These policies take predictions from MultiObjectiveLinUCBAgent.predict_all()
and return the selected arm index.
"""
import numpy as np
from typing import List, Dict, Tuple, Callable


def linear_scalarization_policy(weights: Dict[str, float] = None) -> Callable:
    """
    Creates a policy that selects the arm with the highest weighted sum of UCBs.
    
    Args:
        weights: Dict of objective weights, e.g., {'click': 0.7, 'revenue': 0.3}.
                 Default: Equal weights.
    Returns:
        Policy function.
    """
    def policy(predictions: List[Dict[str, Tuple[float, float]]]) -> int:
        scores = []
        for arm_pred in predictions:
            if weights is None:
                # Equal weight scalarization
                total = sum(pred[1] for pred in arm_pred.values())
            else:
                total = sum(weights.get(obj, 0) * pred[1] for obj, pred in arm_pred.items())
            scores.append(total)
        return int(np.argmax(scores))
    
    return policy


def epsilon_constraint_policy(
    primary_objective: str = 'click',
    constraint_objective: str = 'revenue',
    epsilon: float = 0.3,
    use_conservative_constraint: bool = True
) -> Callable:
    """
    Creates an Epsilon-Constraint policy.
    
    Maximizes the primary objective subject to: constraint_objective >= epsilon.
    
    Args:
        primary_objective: Objective to maximize.
        constraint_objective: Objective to constrain.
        epsilon: Minimum threshold for the constraint objective.
        use_conservative_constraint: If True, use (mean - uncertainty) for constraint check.
    
    Returns:
        Policy function.
    """
    def policy(predictions: List[Dict[str, Tuple[float, float]]]) -> int:
        feasible_arms = []
        fallback_arm = -1
        max_constraint_value = -float('inf')
        
        for arm_idx, arm_pred in enumerate(predictions):
            mean_constraint, ucb_constraint = arm_pred[constraint_objective]
            
            # For constraint checking, be conservative (lower bound)
            if use_conservative_constraint:
                constraint_value = 2 * mean_constraint - ucb_constraint  # mean - alpha*sigma
            else:
                constraint_value = mean_constraint
            
            # Track best fallback (in case no arm is feasible)
            if constraint_value > max_constraint_value:
                max_constraint_value = constraint_value
                fallback_arm = arm_idx
                
            # Check if arm satisfies the constraint
            if constraint_value >= epsilon:
                primary_ucb = arm_pred[primary_objective][1]
                feasible_arms.append((arm_idx, primary_ucb))
        
        if feasible_arms:
            # Return arm with highest primary UCB among feasible
            return max(feasible_arms, key=lambda x: x[1])[0]
        else:
            # Fallback: return arm closest to satisfying constraint
            return fallback_arm
    
    return policy


def pareto_frontier_policy(reference_point: Dict[str, float] = None) -> Callable:
    """
    Creates a Pareto-based policy.
    
    1. Identifies non-dominated arms (Pareto frontier).
    2. Selects using Chebyshev distance to an ideal point (max of all objectives).
    
    Args:
        reference_point: Optional ideal point. If None, computed from predictions.
    
    Returns:
        Policy function.
    """
    def policy(predictions: List[Dict[str, Tuple[float, float]]]) -> int:
        objectives = list(predictions[0].keys())
        n_arms = len(predictions)
        
        # Extract UCB values for all arms
        ucb_matrix = np.array([[arm_pred[obj][1] for obj in objectives] for arm_pred in predictions])
        
        # Find non-dominated arms
        non_dominated_indices = []
        for i in range(n_arms):
            dominated = False
            for j in range(n_arms):
                if i != j:
                    # Check if j dominates i (j >= i in all and j > i in at least one)
                    if np.all(ucb_matrix[j] >= ucb_matrix[i]) and np.any(ucb_matrix[j] > ucb_matrix[i]):
                        dominated = True
                        break
            if not dominated:
                non_dominated_indices.append(i)
        
        if not non_dominated_indices:
            non_dominated_indices = list(range(n_arms))
        
        # Compute ideal point
        if reference_point is None:
            ideal = ucb_matrix.max(axis=0)
        else:
            ideal = np.array([reference_point.get(obj, 1.0) for obj in objectives])
        
        # Select using Chebyshev distance (min-max scalarization)
        best_arm = -1
        min_distance = float('inf')
        for idx in non_dominated_indices:
            # Chebyshev: max over objectives of (ideal - value)
            distance = np.max(ideal - ucb_matrix[idx])
            if distance < min_distance:
                min_distance = distance
                best_arm = idx
        
        return best_arm
    
    return policy
