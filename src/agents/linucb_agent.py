import numpy as np
from .base_agent import BaseAgent
from ..utils.math_utils import sherman_morrison_update

class LinUCBAgent(BaseAgent):
    """
    Implémentation de Disjoint LinUCB avec mise à jour optimisée (Sherman-Morrison).
    """
    
    def __init__(self, n_arms: int, dimension: int, alpha: float = 0.1, lambda_reg: float = 1.0):
        """
        Args:
            alpha (float): Paramètre d'exploration (plus il est grand, plus on explore).
            lambda_reg (float): Régularisation initiale (A = lambda * I).
        """
        super().__init__(n_arms, dimension)
        self.alpha = alpha
        
        # Initialisation des paramètres par bras
        # On stocke directement A_inv pour éviter l'inversion à chaque pas
        # A_inv init à (1/lambda) * I
        self.A_inv = [np.eye(dimension) / lambda_reg for _ in range(n_arms)]
        
        # b init à 0
        self.b = [np.zeros((dimension, 1)) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        # Context shape check (d,) -> (d, 1)
        x = context.reshape(-1, 1)
        
        best_arm = -1
        max_ucb = -float('inf')
        
        for arm in range(self.n_arms):
            A_inv_a = self.A_inv[arm]
            b_a = self.b[arm]
            
            # Estimation theta_hat = A_inv . b
            theta_hat = A_inv_a @ b_a
            
            # Espérance de gain (exploitation)
            # x.T (1, d) @ theta (d, 1) -> scalar
            expected_reward = (x.T @ theta_hat).item()
            
            # Incertitude (exploration)
            # sqrt(x.T . A_inv . x)
            uncertainty = np.sqrt((x.T @ A_inv_a @ x).item())
            
            # UCB Score
            score = expected_reward + self.alpha * uncertainty
            
            if score > max_ucb:
                max_ucb = score
                best_arm = arm
                
        return best_arm

    def update(self, context: np.ndarray, arm: int, reward: float):
        x = context.reshape(-1, 1)
        
        # 1. Mise à jour de A_inv avec Sherman-Morrison
        # A_new = A_old + x x^T  ==> Update A_inv
        self.A_inv[arm] = sherman_morrison_update(self.A_inv[arm], x)
        
        # 2. Mise à jour de b
        # b_new = b_old + r * x
        self.b[arm] += reward * x

    def get_model_params(self):
        return {
            "alpha": self.alpha,
            "n_arms": self.n_arms
        }
