"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) Policy.

Optimized version with:
- Vectorized dominance checking
- Reduced generations for arm selection context
- Efficient crowding distance calculation
"""
import numpy as np
from typing import List, Callable


def nsga2_policy(n_generations: int = 2, use_fast: bool = True) -> Callable:
    """
    Optimized NSGA-II inspired policy for arm selection.
    
    Args:
        n_generations: Number of evolution generations (reduced for speed)
        use_fast: Use fast vectorized operations
    
    Returns:
        Policy function
    """
    def policy(predictions: List[dict]) -> int:
        """
        Select arm using NSGA-II approach.
        """
        n_arms = len(predictions)
        
        # Extract UCB values into numpy array for vectorization
        objectives = np.zeros((n_arms, 2))
        for i, pred in enumerate(predictions):
            objectives[i, 0] = pred['click'][1]  # UCB click
            objectives[i, 1] = pred['revenue'][1]  # UCB revenue
        
        if use_fast:
            # Fast vectorized non-dominated sorting
            pareto_mask = _fast_pareto_front(objectives)
            pareto_indices = np.where(pareto_mask)[0]
        else:
            # Standard approach
            pareto_indices = _get_pareto_indices(objectives)
        
        if len(pareto_indices) == 0:
            return 0
        
        if len(pareto_indices) == 1:
            return pareto_indices[0]
        
        # For multiple Pareto-optimal solutions, use crowding distance
        pareto_objectives = objectives[pareto_indices]
        crowding = _fast_crowding_distance(pareto_objectives)
        
        # Combine crowding with objective values for final selection
        # Prefer high crowding (diversity) AND high total score
        scores = crowding * 0.3 + np.sum(pareto_objectives, axis=1) * 0.7
        
        best_idx = np.argmax(scores)
        return pareto_indices[best_idx]
    
    return policy


def _fast_pareto_front(objectives: np.ndarray) -> np.ndarray:
    """
    Vectorized Pareto front detection.
    
    Args:
        objectives: (N, M) array of objective values
        
    Returns:
        Boolean mask of non-dominated solutions
    """
    n = len(objectives)
    is_dominated = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if is_dominated[i]:
            continue
        # Vectorized dominance check: j dominates i if j >= i in all and j > i in at least one
        better_or_equal = objectives >= objectives[i]  # (N, M)
        strictly_better = objectives > objectives[i]   # (N, M)
        
        # j dominates i if all(better_or_equal[j]) and any(strictly_better[j])
        dominates_i = np.all(better_or_equal, axis=1) & np.any(strictly_better, axis=1)
        dominates_i[i] = False  # Don't compare with self
        
        if np.any(dominates_i):
            is_dominated[i] = True
    
    return ~is_dominated


def _get_pareto_indices(objectives: np.ndarray) -> np.ndarray:
    """Get indices of Pareto-optimal solutions."""
    mask = _fast_pareto_front(objectives)
    return np.where(mask)[0]


def _fast_crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """
    Fast crowding distance calculation using numpy.
    
    Args:
        objectives: (N, M) array of Pareto-optimal solutions
        
    Returns:
        (N,) array of crowding distances
    """
    n, m = objectives.shape
    
    if n <= 2:
        return np.full(n, np.inf)
    
    distances = np.zeros(n)
    
    for obj_idx in range(m):
        # Sort by this objective
        sorted_indices = np.argsort(objectives[:, obj_idx])
        sorted_values = objectives[sorted_indices, obj_idx]
        
        # Boundary solutions get infinite distance
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        # Range for normalization
        obj_range = sorted_values[-1] - sorted_values[0]
        if obj_range == 0:
            continue
        
        # Interior solutions
        for i in range(1, n - 1):
            idx = sorted_indices[i]
            distances[idx] += (sorted_values[i + 1] - sorted_values[i - 1]) / obj_range
    
    return distances


def nsga2_policy_lite() -> Callable:
    """
    Lightweight NSGA-II for real-time use.
    
    Skips evolutionary generations entirely - just finds Pareto front
    and selects using crowding distance.
    """
    def policy(predictions: List[dict]) -> int:
        n_arms = len(predictions)
        
        # Extract objectives
        objectives = np.array([
            [pred['click'][1], pred['revenue'][1]] 
            for pred in predictions
        ])
        
        # Find Pareto front directly
        pareto_mask = _fast_pareto_front(objectives)
        pareto_indices = np.where(pareto_mask)[0]
        
        if len(pareto_indices) <= 1:
            return pareto_indices[0] if len(pareto_indices) == 1 else 0
        
        # Use weighted sum for tie-breaking (fastest)
        pareto_scores = objectives[pareto_indices].sum(axis=1)
        return pareto_indices[np.argmax(pareto_scores)]
    
    return policy
