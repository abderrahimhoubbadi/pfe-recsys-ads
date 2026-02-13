"""
Base Multi-Objective Agent Interface.

All MOO agents must implement predict_all() and return predictions
in the standard format compatible with all MOO policies.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


class BaseMOOAgent(ABC):
    """
    Abstract base class for multi-objective bandit agents.
    
    All agents must return predictions in the format:
        [{'click': (mean, ucb), 'revenue': (mean, ucb)}, ...]  # One dict per arm
    """
    
    def __init__(self, n_arms: int, dimension: int, 
                 objectives: List[str] = None):
        self.n_arms = n_arms
        self.dimension = dimension
        self.objectives = objectives or ['click', 'revenue']
    
    @abstractmethod
    def predict_all(self, context: np.ndarray) -> List[Dict[str, Tuple[float, float]]]:
        """
        For each arm, return predictions for all objectives.
        
        Args:
            context: Feature vector (dimension,)
            
        Returns:
            List of dicts, one per arm:
            [{'click': (mean, upper_bound), 'revenue': (mean, upper_bound)}, ...]
        """
        pass
    
    @abstractmethod
    def update(self, context: np.ndarray, arm: int, rewards: Dict[str, float]):
        """
        Update the agent with observed rewards.
        
        Args:
            context: Feature vector used
            arm: Index of chosen arm
            rewards: Dict of rewards, e.g. {'click': 1, 'revenue': 0.5}
        """
        pass
    
    def select_arm(self, context: np.ndarray, policy: Optional[Callable] = None) -> int:
        """
        Select an arm using a policy function applied to predictions.
        
        Args:
            context: Feature vector
            policy: Callable(predictions) -> int. If None, uses sum of upper bounds.
        """
        predictions = self.predict_all(context)
        
        if policy is None:
            # Default: sum of upper bounds
            scores = [sum(pred[obj][1] for obj in self.objectives) 
                     for pred in predictions]
            return int(np.argmax(scores))
        
        return policy(predictions)
    
    def get_model_params(self) -> Dict:
        return {
            'n_arms': self.n_arms,
            'dimension': self.dimension,
            'objectives': self.objectives,
            'type': self.__class__.__name__
        }
