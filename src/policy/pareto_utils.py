"""
Pareto Utilities for Multi-Objective Optimization.

Provides core functions for Pareto dominance operations.
"""
import numpy as np
from typing import List, Tuple, Dict, Any


def is_dominated(v1: np.ndarray, v2: np.ndarray, maximize: bool = True) -> bool:
    """
    Check if v1 is dominated by v2.
    
    Args:
        v1: First objective vector
        v2: Second objective vector  
        maximize: If True, higher is better; if False, lower is better
        
    Returns:
        True if v1 is dominated by v2 (v2 is better in all objectives)
    """
    v1, v2 = np.asarray(v1), np.asarray(v2)
    
    if maximize:
        # v2 dominates v1 if v2 >= v1 in all and v2 > v1 in at least one
        return np.all(v2 >= v1) and np.any(v2 > v1)
    else:
        # For minimization: v2 dominates if v2 <= v1 in all and v2 < v1 in at least one
        return np.all(v2 <= v1) and np.any(v2 < v1)


def is_pareto_optimal(solution: np.ndarray, all_solutions: List[np.ndarray], 
                       maximize: bool = True) -> bool:
    """
    Check if a solution is Pareto optimal (non-dominated).
    
    Args:
        solution: The solution to check
        all_solutions: List of all solutions
        maximize: If True, higher is better
        
    Returns:
        True if solution is not dominated by any other solution
    """
    for other in all_solutions:
        if not np.array_equal(solution, other):
            if is_dominated(solution, other, maximize):
                return False
    return True


def get_pareto_front(solutions: List[Tuple[int, np.ndarray]], 
                     maximize: bool = True) -> List[Tuple[int, np.ndarray]]:
    """
    Extract the Pareto front from a set of solutions.
    
    Args:
        solutions: List of (arm_index, objective_vector) tuples
        maximize: If True, higher is better
        
    Returns:
        List of non-dominated (arm_index, objective_vector) tuples
    """
    if not solutions:
        return []
    
    pareto_front = []
    
    for i, (arm_i, vec_i) in enumerate(solutions):
        dominated = False
        for j, (arm_j, vec_j) in enumerate(solutions):
            if i != j and is_dominated(vec_i, vec_j, maximize):
                dominated = True
                break
        if not dominated:
            pareto_front.append((arm_i, vec_i))
    
    return pareto_front


def update_pareto_set(current_set: List[Tuple[int, np.ndarray]], 
                      new_solution: Tuple[int, np.ndarray],
                      maximize: bool = True) -> List[Tuple[int, np.ndarray]]:
    """
    Add a new solution to the Pareto set if it's non-dominated.
    
    Args:
        current_set: Current Pareto set
        new_solution: (arm_index, objective_vector) to potentially add
        maximize: If True, higher is better
        
    Returns:
        Updated Pareto set
    """
    arm_new, vec_new = new_solution
    
    # Check if new solution is dominated by any existing
    for _, vec_existing in current_set:
        if is_dominated(vec_new, vec_existing, maximize):
            return current_set  # New solution is dominated, don't add
    
    # Remove solutions dominated by new solution
    updated_set = [
        (arm, vec) for arm, vec in current_set 
        if not is_dominated(vec, vec_new, maximize)
    ]
    
    updated_set.append(new_solution)
    return updated_set


def crowding_distance(front: List[Tuple[int, np.ndarray]]) -> Dict[int, float]:
    """
    Calculate crowding distance for solutions in a Pareto front.
    Used by NSGA-II for diversity preservation.
    
    Args:
        front: List of (arm_index, objective_vector) tuples
        
    Returns:
        Dict mapping arm_index to crowding distance
    """
    if len(front) <= 2:
        return {arm: float('inf') for arm, _ in front}
    
    n_solutions = len(front)
    n_objectives = len(front[0][1])
    
    distances = {arm: 0.0 for arm, _ in front}
    
    for m in range(n_objectives):
        # Sort by objective m
        sorted_front = sorted(front, key=lambda x: x[1][m])
        
        # Boundary solutions get infinite distance
        distances[sorted_front[0][0]] = float('inf')
        distances[sorted_front[-1][0]] = float('inf')
        
        # Calculate range for normalization
        obj_range = sorted_front[-1][1][m] - sorted_front[0][1][m]
        if obj_range == 0:
            continue
        
        # Interior solutions
        for i in range(1, n_solutions - 1):
            arm = sorted_front[i][0]
            distances[arm] += (sorted_front[i+1][1][m] - sorted_front[i-1][1][m]) / obj_range
    
    return distances


def fast_non_dominated_sort(solutions: List[Tuple[int, np.ndarray]], 
                            maximize: bool = True) -> List[List[Tuple[int, np.ndarray]]]:
    """
    Fast non-dominated sorting (NSGA-II).
    
    Args:
        solutions: List of (arm_index, objective_vector) tuples
        maximize: If True, higher is better
        
    Returns:
        List of fronts, where front[0] is the Pareto front
    """
    n = len(solutions)
    domination_count = [0] * n  # Number of solutions that dominate solution i
    dominated_solutions = [[] for _ in range(n)]  # Solutions that solution i dominates
    
    # Calculate domination relations
    for i in range(n):
        for j in range(n):
            if i != j:
                if is_dominated(solutions[j][1], solutions[i][1], maximize):
                    dominated_solutions[i].append(j)
                elif is_dominated(solutions[i][1], solutions[j][1], maximize):
                    domination_count[i] += 1
    
    # Build fronts
    fronts = []
    current_front_indices = [i for i in range(n) if domination_count[i] == 0]
    
    while current_front_indices:
        current_front = [solutions[i] for i in current_front_indices]
        fronts.append(current_front)
        
        next_front_indices = []
        for i in current_front_indices:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front_indices.append(j)
        
        current_front_indices = next_front_indices
    
    return fronts


def chebyshev_distance(point: np.ndarray, reference: np.ndarray, 
                       weights: np.ndarray) -> float:
    """
    Calculate weighted Chebyshev distance (for Pareto frontier policy).
    
    Args:
        point: Objective vector
        reference: Reference point (ideal or nadir)
        weights: Weight vector
        
    Returns:
        Weighted Chebyshev distance
    """
    return np.max(weights * np.abs(point - reference))
