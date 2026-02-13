import numpy as np
from .base_agent import BaseAgent
from ..utils.math_utils import sherman_morrison_update
from typing import Dict, List, Tuple

class MultiObjectiveLinUCBAgent(BaseAgent):
    """
    Multi-Objective LinUCB Agent.
    Learns separate models for each objective (e.g., Click and Revenue) while sharing
    the covariance matrix A (since the context is the same).
    """
    
    def __init__(self, n_arms: int, dimension: int, alpha: float = 0.1, 
                 lambda_reg: float = 1.0, objectives: List[str] = None):
        """
        Args:
            objectives: List of objective names (default: ['click', 'revenue']).
        """
        super().__init__(n_arms, dimension)
        self.alpha = alpha
        self.objectives = objectives or ['click', 'revenue']
        
        # Shared inverse covariance matrix per arm
        self.A_inv = [np.eye(dimension) / lambda_reg for _ in range(n_arms)]
        
        # Separate b vectors for each objective, per arm
        # Structure: {objective_name: [b_arm_0, b_arm_1, ...]}
        self.b = {
            obj: [np.zeros((dimension, 1)) for _ in range(n_arms)]
            for obj in self.objectives
        }

    def predict_all(self, context: np.ndarray) -> List[Dict[str, Tuple[float, float]]]:
        """
        For each arm, return predictions for all objectives.
        
        Returns:
            List of dicts, one per arm: {'click': (mean, ucb), 'revenue': (mean, ucb), ...}
        """
        x = context.reshape(-1, 1)
        predictions = []
        
        for arm in range(self.n_arms):
            A_inv_a = self.A_inv[arm]
            uncertainty = np.sqrt((x.T @ A_inv_a @ x).item())
            
            arm_pred = {}
            for obj in self.objectives:
                b_obj = self.b[obj][arm]
                theta_hat = A_inv_a @ b_obj
                mean = (x.T @ theta_hat).item()
                ucb = mean + self.alpha * uncertainty
                arm_pred[obj] = (mean, ucb)
            
            predictions.append(arm_pred)
        
        return predictions

    def select_arm(self, context: np.ndarray, policy=None) -> int:
        """
        Select an arm using an optional policy function.
        Default: Simple scalarization of all objectives.
        
        Args:
            policy: Callable(predictions) -> int. If None, uses default scalarization.
        """
        predictions = self.predict_all(context)
        
        if policy is None:
            # Default: Equal weight scalarization on UCB
            scores = []
            for arm_pred in predictions:
                total_ucb = sum(pred[1] for pred in arm_pred.values())  # Sum of UCBs
                scores.append(total_ucb)
            return int(np.argmax(scores))
        else:
            return policy(predictions)

    def update(self, context: np.ndarray, arm: int, rewards: Dict[str, float]):
        """
        Update the model with observed rewards for all objectives.
        
        Args:
            rewards: Dict of rewards, e.g., {'click': 1, 'revenue': 0.5}
        """
        x = context.reshape(-1, 1)
        
        # Update shared A_inv with Sherman-Morrison
        self.A_inv[arm] = sherman_morrison_update(self.A_inv[arm], x)
        
        # Update b for each objective
        for obj, reward in rewards.items():
            if obj in self.b:
                self.b[obj][arm] += reward * x

    def get_model_params(self):
        return {
            "alpha": self.alpha,
            "n_arms": self.n_arms,
            "objectives": self.objectives
        }
