"""
Global Semantic LinUCB — A single-model contextual-arm bandit.

Unlike standard LinUCB (which learns one (A, b) pair per arm),
this agent learns ONE global model on the concatenated context:
    x = [user_embedding, ad_embedding]

This enables true Zero-Shot transfer: when a new ad appears,
the model can predict its reward using the global theta, because
the ad embedding captures its semantic meaning relative to known ads.
"""

import numpy as np
from typing import Dict, List, Tuple


class GlobalSemanticLinUCB:
    """
    Multi-Objective LinUCB with a SINGLE global model.

    Context = concat(user_emb, ad_emb), dimension = user_dim + ad_dim.
    One shared A_inv and one b per objective (NOT per arm).
    """

    def __init__(
        self,
        user_dim: int,
        ad_dim: int,
        alpha: float = 0.1,
        objectives: List[str] = None,
    ):
        self.user_dim = user_dim
        self.ad_dim = ad_dim
        self.context_dim = user_dim + ad_dim
        self.alpha = alpha
        self.objectives = objectives or ["click", "revenue"]

        # ONE global inverse covariance matrix
        self.A_inv = np.eye(self.context_dim)

        # ONE b vector per objective
        self.b = {obj: np.zeros((self.context_dim, 1)) for obj in self.objectives}

        # Ad embeddings registry (arm_index -> embedding)
        self.ad_embeddings: Dict[int, np.ndarray] = {}
        self.n_arms = 0

    def set_ad_embeddings(self, embeddings: Dict[int, np.ndarray]):
        """Register ad embeddings for all arms."""
        self.ad_embeddings.update(embeddings)
        self.n_arms = len(self.ad_embeddings)

    def _build_context(self, user_emb: np.ndarray, arm_idx: int) -> np.ndarray:
        """Concatenate user and ad embeddings."""
        ad_emb = self.ad_embeddings.get(arm_idx, np.zeros(self.ad_dim))
        return np.concatenate([user_emb, ad_emb])

    def predict_all(self, user_emb: np.ndarray) -> List[Dict[str, Tuple[float, float]]]:
        """
        For each arm, predict (mean, ucb) for all objectives using the GLOBAL model.
        """
        predictions = []
        for arm in range(self.n_arms):
            ctx = self._build_context(user_emb, arm)
            x = ctx.reshape(-1, 1)

            uncertainty = np.sqrt((x.T @ self.A_inv @ x).item())

            arm_pred = {}
            for obj in self.objectives:
                theta_hat = self.A_inv @ self.b[obj]
                mean = (x.T @ theta_hat).item()
                ucb = mean + self.alpha * uncertainty
                arm_pred[obj] = (mean, ucb)
            predictions.append(arm_pred)

        return predictions

    def select_arm(self, user_emb: np.ndarray, policy=None) -> int:
        """Select an arm using the global model + optional MOO policy."""
        predictions = self.predict_all(user_emb)

        if policy is None:
            scores = [sum(p[1] for p in ap.values()) for ap in predictions]
            return int(np.argmax(scores))
        return policy(predictions)

    def update(self, user_emb: np.ndarray, arm: int, rewards: Dict[str, float]):
        """Update the SINGLE global model with the observed reward."""
        ctx = self._build_context(user_emb, arm)
        x = ctx.reshape(-1, 1)

        # Sherman-Morrison update on the single global A_inv
        Ax = self.A_inv @ x
        denom = 1.0 + (x.T @ Ax).item()
        self.A_inv = self.A_inv - (Ax @ Ax.T) / denom

        # Update b for each objective
        for obj, reward in rewards.items():
            if obj in self.b:
                self.b[obj] += reward * x

    def expand_arms(self, new_embeddings: Dict[int, np.ndarray]):
        """Add new arms with their semantic embeddings. No new matrices needed!"""
        self.ad_embeddings.update(new_embeddings)
        self.n_arms = len(self.ad_embeddings)

    def get_model_params(self) -> dict:
        return {
            "context_dim": self.context_dim,
            "alpha": self.alpha,
            "n_arms": self.n_arms,
            "objectives": self.objectives,
        }
