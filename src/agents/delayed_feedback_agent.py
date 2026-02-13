"""
Delayed Feedback Agent (Multi-Objective).

Handles delayed rewards with bias correction.
In ad systems, clicks/conversions can arrive hours after impression.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from .base_moo_agent import BaseMOOAgent


class DelayedFeedbackAgent(BaseMOOAgent):
    """
    Delayed Feedback Agent (FALCON-inspired).
    
    Maintains two buffers:
    - Pending: impressions with no feedback yet
    - Confirmed: impressions with received feedback
    
    Corrects reward estimates by accounting for the bias introduced
    by delayed feedback (more recent events have lower observed reward).
    
    Ref: Inspired by Ktena et al. (2019) - Addressing Delayed Feedback for
    Continuous Training with Neural Networks in CTR Prediction
    """
    
    def __init__(self, n_arms: int, dimension: int,
                 alpha: float = 0.2, lambda_reg: float = 1.0,
                 delay_window: int = 50,
                 correction_factor: float = 0.8,
                 objectives: List[str] = None):
        """
        Args:
            delay_window: Max pending impressions to track
            correction_factor: Bias correction multiplier for delayed rewards
        """
        super().__init__(n_arms, dimension, objectives)
        self.alpha = alpha
        self.delay_window = delay_window
        self.correction_factor = correction_factor
        
        # Core LinUCB parameters
        self.A_inv = [np.eye(dimension) / lambda_reg for _ in range(n_arms)]
        self.b = {
            obj: [np.zeros(dimension) for _ in range(n_arms)]
            for obj in self.objectives
        }
        
        # Pending impressions buffer
        self.pending = deque(maxlen=delay_window)
        
        # Statistics for correction
        self.confirmed_count = [0] * n_arms
        self.total_count = [0] * n_arms
        self.t = 0
    
    def predict_all(self, context: np.ndarray) -> List[Dict[str, Tuple[float, float]]]:
        x = context.flatten()
        predictions = []
        
        for arm in range(self.n_arms):
            A_inv_a = self.A_inv[arm]
            uncertainty = np.sqrt(x @ A_inv_a @ x) * self.alpha
            
            arm_pred = {}
            for obj in self.objectives:
                theta_hat = A_inv_a @ self.b[obj][arm]
                mean = float(x @ theta_hat)
                
                # Correct for delayed feedback bias
                if self.total_count[arm] > 0:
                    confirm_rate = self.confirmed_count[arm] / self.total_count[arm]
                    # Adjust mean upward if many pending (unconfirmed) events
                    corrected_mean = mean / max(confirm_rate, 0.1)
                else:
                    corrected_mean = mean
                
                ucb = corrected_mean + uncertainty
                arm_pred[obj] = (float(corrected_mean), float(ucb))
            
            predictions.append(arm_pred)
        
        return predictions
    
    def update(self, context: np.ndarray, arm: int, rewards: Dict[str, float]):
        """
        Update with potentially delayed feedback.
        Simulates delay: processes pending impressions probabilistically.
        """
        x = context.flatten()
        self.t += 1
        self.total_count[arm] += 1
        
        # Simulate delayed confirmation
        if np.random.random() < self.correction_factor:
            # Immediate feedback
            self._apply_update(x, arm, rewards)
            self.confirmed_count[arm] += 1
        else:
            # Delayed — add to pending
            self.pending.append((x.copy(), arm, rewards.copy()))
        
        # Process some pending items (simulate delayed arrivals)
        self._process_pending()
    
    def _apply_update(self, x: np.ndarray, arm: int, rewards: Dict[str, float]):
        """Apply LinUCB update."""
        outer = np.outer(x, x)
        A_inv = self.A_inv[arm]
        numerator = A_inv @ outer @ A_inv
        denominator = 1.0 + x @ A_inv @ x
        self.A_inv[arm] = A_inv - numerator / denominator
        
        for obj, reward in rewards.items():
            if obj in self.b:
                # Apply correction factor
                corrected_reward = reward * (1.0 / self.correction_factor)
                self.b[obj][arm] += corrected_reward * x
    
    def _process_pending(self):
        """Process delayed feedback that has now arrived."""
        if not self.pending:
            return
        
        # Process oldest pending items with some probability
        items_to_process = []
        remaining = deque(maxlen=self.delay_window)
        
        for ctx, arm, rewards in self.pending:
            if np.random.random() < 0.3:  # 30% chance of arriving
                items_to_process.append((ctx, arm, rewards))
                self.confirmed_count[arm] += 1
            else:
                remaining.append((ctx, arm, rewards))
        
        self.pending = remaining
        
        for ctx, arm, rewards in items_to_process:
            self._apply_update(ctx, arm, rewards)
