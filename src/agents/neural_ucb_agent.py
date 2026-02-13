"""
NeuralUCB Agent (Multi-Objective) — Optimized.

Uses CPU for fast single-sample inference + batch GPU training.
GPU kernel launch overhead is too high for tiny single-sample ops,
so we accumulate a buffer and train in batches on GPU.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from .base_moo_agent import BaseMOOAgent

GPU_AVAILABLE = torch.cuda.is_available()
TRAIN_DEVICE = torch.device('cuda' if GPU_AVAILABLE else 'cpu')


class RewardNetwork(nn.Module):
    """Small MLP for reward prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralUCBAgent(BaseMOOAgent):
    """
    NeuralUCB: Neural Contextual Bandits with UCB Exploration.
    
    Optimized: CPU inference + GPU batch training.
    - Inference on CPU (tiny model, no kernel overhead)
    - Accumulates data, trains in GPU batches
    
    Ref: Zhou et al. (2020)
    """
    
    def __init__(self, n_arms: int, dimension: int,
                 alpha: float = 0.1, hidden_dim: int = 32,
                 lr: float = 0.01, lambda_reg: float = 1.0,
                 batch_size: int = 16,
                 objectives: List[str] = None):
        super().__init__(n_arms, dimension, objectives)
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Networks stay on CPU for fast inference
        self.networks = {}
        self.optimizers = {}
        
        for obj in self.objectives:
            self.networks[obj] = [
                RewardNetwork(dimension, hidden_dim) for _ in range(n_arms)
            ]
            self.optimizers[obj] = [
                optim.SGD(self.networks[obj][a].parameters(), lr=lr)
                for a in range(n_arms)
            ]
        
        # Data buffers for batch training
        self.buffers = {
            obj: {a: {'x': [], 'y': []} for a in range(n_arms)}
            for obj in self.objectives
        }
        self.t = 0
    
    def predict_all(self, context: np.ndarray) -> List[Dict[str, Tuple[float, float]]]:
        x = torch.FloatTensor(context.flatten())
        predictions = []
        
        for arm in range(self.n_arms):
            arm_pred = {}
            for obj in self.objectives:
                net = self.networks[obj][arm]
                net.eval()
                
                # Fast CPU inference
                with torch.no_grad():
                    mean = net(x).item()
                
                # Simplified uncertainty: gradient norm
                x_grad = x.clone().requires_grad_(True)
                net.zero_grad()
                out = net(x_grad)
                out.backward()
                grads = []
                for p in net.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.detach().flatten())
                grad_norm = torch.cat(grads).norm().item() if grads else 0.0
                uncertainty = grad_norm * self.alpha
                
                arm_pred[obj] = (float(mean), float(mean + uncertainty))
            
            predictions.append(arm_pred)
        
        return predictions
    
    def update(self, context: np.ndarray, arm: int, rewards: Dict[str, float]):
        x = context.flatten()
        self.t += 1
        
        for obj, reward in rewards.items():
            if obj not in self.buffers:
                continue
            self.buffers[obj][arm]['x'].append(x.copy())
            self.buffers[obj][arm]['y'].append(float(reward))
            
            # Batch train when buffer is full
            if len(self.buffers[obj][arm]['x']) >= self.batch_size:
                self._batch_train(obj, arm)
    
    def _batch_train(self, obj: str, arm: int):
        """Train on accumulated batch, using GPU if available."""
        buf = self.buffers[obj][arm]
        if not buf['x']:
            return
        
        X = torch.FloatTensor(np.array(buf['x']))
        Y = torch.FloatTensor(buf['y'])
        
        net = self.networks[obj][arm]
        optimizer = self.optimizers[obj][arm]
        
        if GPU_AVAILABLE:
            # Move batch to GPU, train, move back
            net.to(TRAIN_DEVICE)
            X, Y = X.to(TRAIN_DEVICE), Y.to(TRAIN_DEVICE)
        
        net.train()
        optimizer.zero_grad()
        preds = net(X)
        loss = ((preds - Y) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        if GPU_AVAILABLE:
            net.to('cpu')
        
        # Clear buffer
        buf['x'].clear()
        buf['y'].clear()
