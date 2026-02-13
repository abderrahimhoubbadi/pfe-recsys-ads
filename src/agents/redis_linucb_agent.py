import numpy as np
import logging
from .base_agent import BaseAgent
from ..infra.redis_client import RedisClient
from ..utils.math_utils import sherman_morrison_update

class RedisLinUCBAgent(BaseAgent):
    """
    Stateless implementation of LinUCB that persists state in Redis.
    Uses Sherman-Morrison for updates.
    """
    
    def __init__(self, agent_id: str, n_arms: int, dimension: int, 
                 redis_client: RedisClient, alpha: float = 0.1, lambda_reg: float = 1.0):
        """
        Args:
            agent_id (str): Unique identifier for this agent in Redis.
            n_arms (int): Number of arms.
            dimension (int): Dimension of context vectors.
            redis_client (RedisClient): Instance of Redis wrapper.
            alpha (float): Exploration parameter.
            lambda_reg (float): Initial regularization.
        """
        super().__init__(n_arms, dimension)
        self.agent_id = agent_id
        self.redis = redis_client
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.logger = logging.getLogger(__name__)

    def _get_arm_id(self, arm_index: int) -> str:
        return f"{self.agent_id}:arm:{arm_index}"

    def _fetch_or_init_model(self, arm_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Fetches (A_inv, b) from Redis or initializes them if missing."""
        arm_id = self._get_arm_id(arm_index)
        A_inv, b = self.redis.get_model(arm_id, self.dimension)

        if A_inv is None or b is None:
            # Initialize if not exists
            A_inv = np.eye(self.dimension) / self.lambda_reg
            b = np.zeros((self.dimension, 1))
            # Save immediately to ensure consistency
            self.redis.update_model(arm_id, A_inv, b)
        
        return A_inv, b

    def select_arm(self, context: np.ndarray) -> int:
        x = context.reshape(-1, 1)
        best_arm = -1
        max_ucb = -float('inf')

        for arm in range(self.n_arms):
            A_inv, b = self._fetch_or_init_model(arm)
            
            # Estimator theta_hat = A_inv . b
            theta_hat = A_inv @ b
            
            # Exploitation
            expected_reward = (x.T @ theta_hat).item()
            
            # Exploration
            uncertainty = np.sqrt((x.T @ A_inv @ x).item())
            
            # UCB
            score = expected_reward + self.alpha * uncertainty
            
            if score > max_ucb:
                max_ucb = score
                best_arm = arm
                
        return best_arm

    def update(self, context: np.ndarray, arm: int, reward: float):
        x = context.reshape(-1, 1)
        arm_id = self._get_arm_id(arm)
        
        # Fetch current state
        # Note: Race condition possible here in distributed setting without locking
        # For V1 we assume single writer or accept optimistic concurrency issues
        A_inv, b = self._fetch_or_init_model(arm)
        
        # Update A_inv with Sherman-Morrison
        A_inv = sherman_morrison_update(A_inv, x)
        
        # Update b
        b += reward * x
        
        # Save back to Redis
        self.redis.update_model(arm_id, A_inv, b)

    def get_model_params(self):
        return {
            "agent_id": self.agent_id,
            "alpha": self.alpha,
            "lambda_reg": self.lambda_reg
        }
