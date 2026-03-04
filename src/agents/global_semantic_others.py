"""
Global Semantic Linear Agents — Thompson, OfflineOnline, DelayedFeedback.

Each agent uses ONE global linear model on concat(user_emb, ad_emb)
instead of K per-arm models. Enables zero-shot generalization.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import deque


# ================================================================
# Global Semantic Thompson Sampling
# ================================================================


class GlobalSemanticThompson:
    """
    Thompson Sampling with ONE global Bayesian linear model.
    Context = concat(user_emb, ad_emb). Samples θ ~ N(θ_hat, v² A_inv).
    """

    def __init__(
        self,
        user_dim: int,
        ad_dim: int,
        v: float = 0.2,
        lambda_reg: float = 1.0,
        objectives: List[str] = None,
    ):
        self.user_dim = user_dim
        self.ad_dim = ad_dim
        self.context_dim = user_dim + ad_dim
        self.v = v
        self.objectives = objectives or ["click", "revenue"]

        # ONE global inverse covariance (no need to store A with Sherman-Morrison)
        self.A_inv = np.eye(self.context_dim) / lambda_reg

        # ONE b per objective
        self.b = {obj: np.zeros(self.context_dim) for obj in self.objectives}

        self.ad_embeddings: Dict[int, np.ndarray] = {}
        self.n_arms = 0

    def set_ad_embeddings(self, embeddings: Dict[int, np.ndarray]):
        self.ad_embeddings.update(embeddings)
        self.n_arms = len(self.ad_embeddings)

    def _build_context(self, user_emb: np.ndarray, arm: int) -> np.ndarray:
        ad_emb = self.ad_embeddings.get(arm, np.zeros(self.ad_dim))
        return np.concatenate([user_emb, ad_emb])

    def select_arm(self, user_emb: np.ndarray, policy=None) -> int:
        predictions = []
        for arm in range(self.n_arms):
            ctx = self._build_context(user_emb, arm)
            arm_pred = {}
            for obj in self.objectives:
                theta_hat = self.A_inv @ self.b[obj]
                mean = float(ctx @ theta_hat)

                # Math shortcut: x^T theta ~ N(x^T theta_hat, v^2 x^T A_inv x)
                # Avoids O(d^3) 768-dimensional multivariate sampling
                variance = (self.v**2) * (ctx @ self.A_inv @ ctx)
                sampled = float(np.random.normal(mean, np.sqrt(variance)))

                arm_pred[obj] = (mean, sampled)
            predictions.append(arm_pred)

        if policy is None:
            scores = [sum(p[1] for p in ap.values()) for ap in predictions]
            return int(np.argmax(scores))
        return policy(predictions)

    def update(self, user_emb: np.ndarray, arm: int, rewards: Dict[str, float]):
        ctx = self._build_context(user_emb, arm)

        # Sherman-Morrison rank-1 update: O(d^2) instead of O(d^3)
        Ax = self.A_inv @ ctx
        denom = 1.0 + ctx @ Ax
        self.A_inv -= np.outer(Ax, Ax) / denom

        for obj, reward in rewards.items():
            if obj in self.b:
                self.b[obj] += reward * ctx

    def expand_arms(self, new_embeddings: Dict[int, np.ndarray]):
        self.ad_embeddings.update(new_embeddings)
        self.n_arms = len(self.ad_embeddings)


# ================================================================
# Global Semantic Offline-to-Online
# ================================================================


class GlobalSemanticOfflineOnline:
    """
    Offline-to-Online with ONE global LinUCB model.
    Pessimism decays to optimism over time.
    """

    def __init__(
        self,
        user_dim: int,
        ad_dim: int,
        alpha: float = 0.2,
        lambda_reg: float = 1.0,
        pessimism_decay: float = 0.995,
        objectives: List[str] = None,
    ):
        self.user_dim = user_dim
        self.ad_dim = ad_dim
        self.context_dim = user_dim + ad_dim
        self.alpha = alpha
        self.pessimism_decay = pessimism_decay
        self.pessimism_weight = 1.0
        self.objectives = objectives or ["click", "revenue"]

        self.A_inv = np.eye(self.context_dim) / lambda_reg
        self.b = {obj: np.zeros((self.context_dim, 1)) for obj in self.objectives}

        self.ad_embeddings: Dict[int, np.ndarray] = {}
        self.n_arms = 0

    def set_ad_embeddings(self, embeddings: Dict[int, np.ndarray]):
        self.ad_embeddings.update(embeddings)
        self.n_arms = len(self.ad_embeddings)

    def _build_context(self, user_emb: np.ndarray, arm: int) -> np.ndarray:
        ad_emb = self.ad_embeddings.get(arm, np.zeros(self.ad_dim))
        return np.concatenate([user_emb, ad_emb])

    def select_arm(self, user_emb: np.ndarray, policy=None) -> int:
        predictions = []
        for arm in range(self.n_arms):
            ctx = self._build_context(user_emb, arm)
            x = ctx.reshape(-1, 1)
            uncertainty = np.sqrt((x.T @ self.A_inv @ x).item()) * self.alpha

            arm_pred = {}
            for obj in self.objectives:
                theta_hat = self.A_inv @ self.b[obj]
                mean = (x.T @ theta_hat).item()
                ucb = mean + uncertainty
                arm_pred[obj] = (float(mean), float(ucb))
            predictions.append(arm_pred)

        if policy is None:
            scores = [sum(p[1] for p in ap.values()) for ap in predictions]
            return int(np.argmax(scores))
        return policy(predictions)

    def update(self, user_emb: np.ndarray, arm: int, rewards: Dict[str, float]):
        ctx = self._build_context(user_emb, arm)
        x = ctx.reshape(-1, 1)
        self.pessimism_weight *= self.pessimism_decay

        # Sherman-Morrison
        Ax = self.A_inv @ x
        denom = 1.0 + (x.T @ Ax).item()
        self.A_inv = self.A_inv - (Ax @ Ax.T) / denom

        for obj, reward in rewards.items():
            if obj in self.b:
                self.b[obj] += reward * x

    def expand_arms(self, new_embeddings: Dict[int, np.ndarray]):
        self.ad_embeddings.update(new_embeddings)
        self.n_arms = len(self.ad_embeddings)


# ================================================================
# Global Semantic Delayed Feedback
# ================================================================


class GlobalSemanticDelayedFeedback:
    """
    Delayed Feedback with ONE global LinUCB model.
    Simulates delayed reward arrival with bias correction.
    """

    def __init__(
        self,
        user_dim: int,
        ad_dim: int,
        alpha: float = 0.2,
        lambda_reg: float = 1.0,
        delay_window: int = 50,
        correction_factor: float = 0.8,
        objectives: List[str] = None,
    ):
        self.user_dim = user_dim
        self.ad_dim = ad_dim
        self.context_dim = user_dim + ad_dim
        self.alpha = alpha
        self.correction_factor = correction_factor
        self.objectives = objectives or ["click", "revenue"]

        self.A_inv = np.eye(self.context_dim) / lambda_reg
        self.b = {obj: np.zeros((self.context_dim, 1)) for obj in self.objectives}

        self.pending = deque(maxlen=delay_window)
        self.confirmed_count = 0
        self.total_count = 0

        self.ad_embeddings: Dict[int, np.ndarray] = {}
        self.n_arms = 0

    def set_ad_embeddings(self, embeddings: Dict[int, np.ndarray]):
        self.ad_embeddings.update(embeddings)
        self.n_arms = len(self.ad_embeddings)

    def _build_context(self, user_emb: np.ndarray, arm: int) -> np.ndarray:
        ad_emb = self.ad_embeddings.get(arm, np.zeros(self.ad_dim))
        return np.concatenate([user_emb, ad_emb])

    def select_arm(self, user_emb: np.ndarray, policy=None) -> int:
        predictions = []
        for arm in range(self.n_arms):
            ctx = self._build_context(user_emb, arm)
            x = ctx.reshape(-1, 1)
            uncertainty = np.sqrt((x.T @ self.A_inv @ x).item()) * self.alpha

            arm_pred = {}
            for obj in self.objectives:
                theta_hat = self.A_inv @ self.b[obj]
                mean = (x.T @ theta_hat).item()

                if self.total_count > 0:
                    confirm_rate = self.confirmed_count / self.total_count
                    corrected_mean = mean / max(confirm_rate, 0.1)
                else:
                    corrected_mean = mean

                ucb = corrected_mean + uncertainty
                arm_pred[obj] = (float(corrected_mean), float(ucb))
            predictions.append(arm_pred)

        if policy is None:
            scores = [sum(p[1] for p in ap.values()) for ap in predictions]
            return int(np.argmax(scores))
        return policy(predictions)

    def update(self, user_emb: np.ndarray, arm: int, rewards: Dict[str, float]):
        ctx = self._build_context(user_emb, arm)
        self.total_count += 1

        if np.random.random() < self.correction_factor:
            self._apply_update(ctx, rewards)
            self.confirmed_count += 1
        else:
            self.pending.append((ctx.copy(), rewards.copy()))

        self._process_pending()

    def _apply_update(self, ctx: np.ndarray, rewards: Dict[str, float]):
        x = ctx.reshape(-1, 1)
        Ax = self.A_inv @ x
        denom = 1.0 + (x.T @ Ax).item()
        self.A_inv = self.A_inv - (Ax @ Ax.T) / denom

        for obj, reward in rewards.items():
            if obj in self.b:
                corrected_reward = reward * (1.0 / self.correction_factor)
                self.b[obj] += corrected_reward * x

    def _process_pending(self):
        if not self.pending:
            return
        remaining = deque(maxlen=self.pending.maxlen)
        for ctx, rewards in self.pending:
            if np.random.random() < 0.3:
                self._apply_update(ctx, rewards)
                self.confirmed_count += 1
            else:
                remaining.append((ctx, rewards))
        self.pending = remaining

    def expand_arms(self, new_embeddings: Dict[int, np.ndarray]):
        self.ad_embeddings.update(new_embeddings)
        self.n_arms = len(self.ad_embeddings)
