"""
Global Semantic Neural Agents — NeuralUCB, NeuralTS, DeepBandit.

Each agent uses ONE global neural network on concat(user_emb, ad_emb)
instead of K separate per-arm networks. This enables zero-shot
generalization to new ads via their semantic embeddings.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple

GPU_AVAILABLE = torch.cuda.is_available()
TRAIN_DEVICE = torch.device("cuda" if GPU_AVAILABLE else "cpu")


class GlobalMLP(nn.Module):
    """Shared MLP for all arms (input = user_emb + ad_emb)."""

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ================================================================
# Global Semantic NeuralUCB
# ================================================================


class GlobalSemanticNeuralUCB:
    """
    NeuralUCB with ONE global network for all arms.
    Context = concat(user_emb, ad_emb). Exploration = gradient norm UCB.
    """

    def __init__(
        self,
        user_dim: int,
        ad_dim: int,
        alpha: float = 0.1,
        hidden_dim: int = 32,
        lr: float = 0.01,
        batch_size: int = 16,
        objectives: List[str] = None,
    ):
        self.user_dim = user_dim
        self.ad_dim = ad_dim
        self.context_dim = user_dim + ad_dim
        self.alpha = alpha
        self.batch_size = batch_size
        self.objectives = objectives or ["click", "revenue"]

        # ONE network per objective (shared across all arms)
        self.networks = {
            obj: GlobalMLP(self.context_dim, hidden_dim) for obj in self.objectives
        }
        self.optimizers = {
            obj: optim.Adam(self.networks[obj].parameters(), lr=lr)
            for obj in self.objectives
        }

        # Single global data buffer per objective
        self.buffers = {obj: {"x": [], "y": []} for obj in self.objectives}

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
            x = torch.FloatTensor(ctx)
            arm_pred = {}
            for obj in self.objectives:
                net = self.networks[obj]
                net.eval()
                with torch.no_grad():
                    mean = net(x).item()

                x_grad = x.clone().requires_grad_(True)
                net.zero_grad()
                out = net(x_grad)
                out.backward()
                grads = [
                    p.grad.detach().flatten()
                    for p in net.parameters()
                    if p.grad is not None
                ]
                grad_norm = torch.cat(grads).norm().item() if grads else 0.0
                uncertainty = grad_norm * self.alpha
                arm_pred[obj] = (float(mean), float(mean + uncertainty))
            predictions.append(arm_pred)

        if policy is None:
            scores = [sum(p[1] for p in ap.values()) for ap in predictions]
            return int(np.argmax(scores))
        return policy(predictions)

    def update(self, user_emb: np.ndarray, arm: int, rewards: Dict[str, float]):
        ctx = self._build_context(user_emb, arm)
        for obj, reward in rewards.items():
            if obj not in self.buffers:
                continue
            self.buffers[obj]["x"].append(ctx.copy())
            self.buffers[obj]["y"].append(float(reward))
            if len(self.buffers[obj]["x"]) >= self.batch_size:
                self._batch_train(obj)

    def _batch_train(self, obj: str):
        buf = self.buffers[obj]
        if not buf["x"]:
            return
        X = torch.FloatTensor(np.array(buf["x"]))
        Y = torch.FloatTensor(buf["y"])
        net = self.networks[obj]
        optimizer = self.optimizers[obj]

        if GPU_AVAILABLE:
            net.to(TRAIN_DEVICE)
            X, Y = X.to(TRAIN_DEVICE), Y.to(TRAIN_DEVICE)

        net.train()
        optimizer.zero_grad()
        preds = net(X)
        loss = ((preds - Y) ** 2).mean()
        loss.backward()
        optimizer.step()

        if GPU_AVAILABLE:
            net.to("cpu")
        buf["x"].clear()
        buf["y"].clear()

    def expand_arms(self, new_embeddings: Dict[int, np.ndarray]):
        self.ad_embeddings.update(new_embeddings)
        self.n_arms = len(self.ad_embeddings)


# ================================================================
# Global Semantic NeuralTS
# ================================================================


class GlobalSemanticNeuralTS:
    """
    NeuralTS with ONE global network. Exploration via parameter perturbation.
    Context = concat(user_emb, ad_emb).
    """

    def __init__(
        self,
        user_dim: int,
        ad_dim: int,
        sigma: float = 0.05,
        hidden_dim: int = 32,
        lr: float = 0.01,
        batch_size: int = 16,
        objectives: List[str] = None,
    ):
        self.user_dim = user_dim
        self.ad_dim = ad_dim
        self.context_dim = user_dim + ad_dim
        self.sigma = sigma
        self.batch_size = batch_size
        self.objectives = objectives or ["click", "revenue"]

        self.networks = {
            obj: GlobalMLP(self.context_dim, hidden_dim) for obj in self.objectives
        }
        self.optimizers = {
            obj: optim.Adam(self.networks[obj].parameters(), lr=lr)
            for obj in self.objectives
        }
        self.buffers = {obj: {"x": [], "y": []} for obj in self.objectives}

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
            x = torch.FloatTensor(ctx)
            arm_pred = {}
            for obj in self.objectives:
                net = self.networks[obj]
                net.eval()
                with torch.no_grad():
                    mean = net(x).item()

                # Thompson sample via parameter perturbation
                original_params = [p.data.clone() for p in net.parameters()]
                with torch.no_grad():
                    for p in net.parameters():
                        p.data += torch.randn_like(p) * self.sigma
                    sampled = net(x).item()
                    for p, orig in zip(net.parameters(), original_params):
                        p.data.copy_(orig)

                arm_pred[obj] = (float(mean), float(sampled))
            predictions.append(arm_pred)

        if policy is None:
            scores = [sum(p[1] for p in ap.values()) for ap in predictions]
            return int(np.argmax(scores))
        return policy(predictions)

    def update(self, user_emb: np.ndarray, arm: int, rewards: Dict[str, float]):
        ctx = self._build_context(user_emb, arm)
        for obj, reward in rewards.items():
            if obj not in self.buffers:
                continue
            self.buffers[obj]["x"].append(ctx.copy())
            self.buffers[obj]["y"].append(float(reward))
            if len(self.buffers[obj]["x"]) >= self.batch_size:
                self._batch_train(obj)

    def _batch_train(self, obj: str):
        buf = self.buffers[obj]
        if not buf["x"]:
            return
        X = torch.FloatTensor(np.array(buf["x"]))
        Y = torch.FloatTensor(buf["y"])
        net = self.networks[obj]
        optimizer = self.optimizers[obj]
        if GPU_AVAILABLE:
            net.to(TRAIN_DEVICE)
            X, Y = X.to(TRAIN_DEVICE), Y.to(TRAIN_DEVICE)
        net.train()
        optimizer.zero_grad()
        preds = net(X)
        loss = ((preds - Y) ** 2).mean()
        loss.backward()
        optimizer.step()
        if GPU_AVAILABLE:
            net.to("cpu")
        buf["x"].clear()
        buf["y"].clear()

    def expand_arms(self, new_embeddings: Dict[int, np.ndarray]):
        self.ad_embeddings.update(new_embeddings)
        self.n_arms = len(self.ad_embeddings)


# ================================================================
# Global Semantic DeepBandit (Bootstrap Ensemble)
# ================================================================


class GlobalSemanticDeepBandit:
    """
    DeepBandit with ONE global ensemble (shared across all arms).
    Each ensemble member is a GlobalMLP on concat(user_emb, ad_emb).
    Uncertainty = std across ensemble predictions.
    """

    def __init__(
        self,
        user_dim: int,
        ad_dim: int,
        alpha: float = 1.0,
        n_ensemble: int = 5,
        hidden_dim: int = 32,
        lr: float = 0.01,
        batch_size: int = 16,
        objectives: List[str] = None,
    ):
        self.user_dim = user_dim
        self.ad_dim = ad_dim
        self.context_dim = user_dim + ad_dim
        self.alpha = alpha
        self.n_ensemble = n_ensemble
        self.batch_size = batch_size
        self.objectives = objectives or ["click", "revenue"]

        # ONE global ensemble per objective
        self.ensembles = {
            obj: [GlobalMLP(self.context_dim, hidden_dim) for _ in range(n_ensemble)]
            for obj in self.objectives
        }
        self.optimizers = {
            obj: [
                optim.Adam(self.ensembles[obj][e].parameters(), lr=lr)
                for e in range(n_ensemble)
            ]
            for obj in self.objectives
        }
        self.buffers = {obj: {"x": [], "y": [], "mask": []} for obj in self.objectives}

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
            x = torch.FloatTensor(ctx)
            arm_pred = {}
            for obj in self.objectives:
                preds = []
                with torch.no_grad():
                    for net in self.ensembles[obj]:
                        net.eval()
                        preds.append(net(x).item())
                mean = float(np.mean(preds))
                std = float(np.std(preds))
                arm_pred[obj] = (mean, mean + self.alpha * std)
            predictions.append(arm_pred)

        if policy is None:
            scores = [sum(p[1] for p in ap.values()) for ap in predictions]
            return int(np.argmax(scores))
        return policy(predictions)

    def update(self, user_emb: np.ndarray, arm: int, rewards: Dict[str, float]):
        ctx = self._build_context(user_emb, arm)
        mask = [1 if np.random.random() < 0.7 else 0 for _ in range(self.n_ensemble)]
        for obj, reward in rewards.items():
            if obj not in self.buffers:
                continue
            self.buffers[obj]["x"].append(ctx.copy())
            self.buffers[obj]["y"].append(float(reward))
            self.buffers[obj]["mask"].append(mask.copy())
            if len(self.buffers[obj]["x"]) >= self.batch_size:
                self._batch_train(obj)

    def _batch_train(self, obj: str):
        buf = self.buffers[obj]
        if not buf["x"]:
            return
        X = torch.FloatTensor(np.array(buf["x"]))
        Y = torch.FloatTensor(buf["y"])
        masks = np.array(buf["mask"])

        for e in range(self.n_ensemble):
            sample_mask = masks[:, e].astype(bool)
            if not np.any(sample_mask):
                continue
            X_e, Y_e = X[sample_mask], Y[sample_mask]
            net = self.ensembles[obj][e]
            optimizer = self.optimizers[obj][e]
            if GPU_AVAILABLE:
                net.to(TRAIN_DEVICE)
                X_e, Y_e = X_e.to(TRAIN_DEVICE), Y_e.to(TRAIN_DEVICE)
            net.train()
            optimizer.zero_grad()
            preds = net(X_e)
            loss = ((preds - Y_e) ** 2).mean()
            loss.backward()
            optimizer.step()
            if GPU_AVAILABLE:
                net.to("cpu")

        buf["x"].clear()
        buf["y"].clear()
        buf["mask"].clear()

    def expand_arms(self, new_embeddings: Dict[int, np.ndarray]):
        self.ad_embeddings.update(new_embeddings)
        self.n_arms = len(self.ad_embeddings)
