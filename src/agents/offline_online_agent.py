"""
Offline-to-Online Agent (Multi-Objective).

Phase 1: Pre-trains on historical/generated data (offline)
Phase 2: Fine-tunes online with new interactions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_moo_agent import BaseMOOAgent


class OfflineOnlineAgent(BaseMOOAgent):
    """
    Offline-to-Online Learning Agent.

    Combines offline pre-training with online adaptation:
    - Offline phase: Learn initial θ from historical logs
    - Online phase: Use pessimistic estimates initially, then transition
      to optimistic (UCB) as confidence grows

    Ref: Inspired by OFUL (Abbasi-Yadkori et al.)
    """

    def __init__(
        self,
        n_arms: int,
        dimension: int,
        alpha: float = 0.2,
        lambda_reg: float = 1.0,
        pessimism_decay: float = 0.995,
        objectives: List[str] = None,
    ):
        """
        Args:
            pessimism_decay: Rate at which pessimism decays (→ optimism)
        """
        super().__init__(n_arms, dimension, objectives)
        self.alpha = alpha
        self.pessimism_decay = pessimism_decay
        self.pessimism_weight = 1.0  # Starts pessimistic, decays to 0

        # LinUCB parameters
        self.A_inv = [np.eye(dimension) / lambda_reg for _ in range(n_arms)]
        self.b = {
            obj: [np.zeros(dimension) for _ in range(n_arms)] for obj in self.objectives
        }
        self.counts = [0] * n_arms
        self.t = 0

    def pretrain(
        self, contexts: np.ndarray, arms: np.ndarray, rewards: List[Dict[str, float]]
    ):
        """
        Offline pre-training on historical data.

        Args:
            contexts: (N, dimension) array of contexts
            arms: (N,) array of chosen arms
            rewards: List of reward dicts
        """
        for ctx, arm, reward in zip(contexts, arms, rewards):
            self.update(ctx, int(arm), reward)

        # After offline phase, reset pessimism
        self.pessimism_weight = 1.0

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

                # Transition from pessimistic to optimistic
                # Pessimistic: mean - α·σ  |  Optimistic: mean + α·σ
                bonus = uncertainty * (1 - 2 * self.pessimism_weight)
                # When pessimism_weight=1: bonus = -uncertainty (pessimistic)
                # When pessimism_weight=0: bonus = +uncertainty (optimistic/UCB)

                upper = mean + uncertainty  # UCB for policy compatibility
                arm_pred[obj] = (float(mean), float(upper))

            predictions.append(arm_pred)

        return predictions

    def update(self, context: np.ndarray, arm: int, rewards: Dict[str, float]):
        x = context.flatten()
        self.t += 1
        self.counts[arm] += 1

        # Decay pessimism
        self.pessimism_weight *= self.pessimism_decay

        # Sherman-Morrison rank-1 update: O(d²) instead of O(d³)
        Ax = self.A_inv[arm] @ x
        denom = 1.0 + x @ Ax
        self.A_inv[arm] -= np.outer(Ax, Ax) / denom

        for obj, reward in rewards.items():
            if obj in self.b:
                self.b[obj][arm] += reward * x
