"""
Linear Thompson Sampling Agent (Multi-Objective).

Instead of UCB bounds, samples parameter vectors from the posterior
distribution for exploration. More natural Bayesian exploration.
"""
import numpy as np
from typing import Dict, List, Tuple
from .base_moo_agent import BaseMOOAgent


class ThompsonSamplingAgent(BaseMOOAgent):
    """
    Multi-Objective Linear Thompson Sampling.
    
    Maintains a Gaussian posterior over θ for each (arm, objective).
    At decision time, samples θ ~ N(θ_hat, v² * A_inv) and uses the
    sampled value as the exploration bonus.
    """
    
    def __init__(self, n_arms: int, dimension: int, 
                 v: float = 0.2, lambda_reg: float = 1.0,
                 objectives: List[str] = None):
        """
        Args:
            v: Controls exploration variance (similar to alpha in LinUCB)
            lambda_reg: Regularization parameter
        """
        super().__init__(n_arms, dimension, objectives)
        self.v = v
        
        # Per-arm shared covariance
        self.A = [lambda_reg * np.eye(dimension) for _ in range(n_arms)]
        self.A_inv = [np.eye(dimension) / lambda_reg for _ in range(n_arms)]
        
        # Per-arm per-objective reward accumulator
        self.b = {
            obj: [np.zeros(dimension) for _ in range(n_arms)]
            for obj in self.objectives
        }
    
    def predict_all(self, context: np.ndarray) -> List[Dict[str, Tuple[float, float]]]:
        """
        Sample θ from posterior and compute predictions.
        Returns (mean, sampled_value) per objective.
        """
        x = context.flatten()
        predictions = []
        
        for arm in range(self.n_arms):
            A_inv_a = self.A_inv[arm]
            arm_pred = {}
            
            for obj in self.objectives:
                # Posterior mean
                theta_hat = A_inv_a @ self.b[obj][arm]
                mean = x @ theta_hat
                
                # Sample from posterior: θ ~ N(θ_hat, v² * A_inv)
                theta_sample = np.random.multivariate_normal(
                    theta_hat, self.v**2 * A_inv_a
                )
                sampled_value = x @ theta_sample
                
                arm_pred[obj] = (float(mean), float(sampled_value))
            
            predictions.append(arm_pred)
        
        return predictions
    
    def update(self, context: np.ndarray, arm: int, rewards: Dict[str, float]):
        x = context.flatten()
        
        # Update A and A_inv (Sherman-Morrison)
        outer = np.outer(x, x)
        self.A[arm] += outer
        self.A_inv[arm] = np.linalg.solve(self.A[arm], np.eye(self.dimension))
        
        # Update b for each objective
        for obj, reward in rewards.items():
            if obj in self.b:
                self.b[obj][arm] += reward * x
