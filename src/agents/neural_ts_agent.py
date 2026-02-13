"""
Neural Thompson Sampling Agent (Multi-Objective) — Optimized.

CPU inference + GPU batch training.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from .base_moo_agent import BaseMOOAgent

GPU_AVAILABLE = torch.cuda.is_available()
TRAIN_DEVICE = torch.device('cuda' if GPU_AVAILABLE else 'cpu')


class NeuralTSNetwork(nn.Module):
    """MLP with noise injection for Thompson Sampling."""
    
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


class NeuralTSAgent(BaseMOOAgent):
    """
    Neural Thompson Sampling — Optimized.
    
    CPU inference (fast per-sample) + GPU batch training.
    Exploration via parameter perturbation.
    
    Ref: Zhang et al. (2021)
    """
    
    def __init__(self, n_arms: int, dimension: int,
                 sigma: float = 0.05, hidden_dim: int = 32,
                 lr: float = 0.01, batch_size: int = 16,
                 objectives: List[str] = None):
        super().__init__(n_arms, dimension, objectives)
        self.sigma = sigma
        self.batch_size = batch_size
        
        # Networks on CPU for fast inference
        self.networks = {}
        self.optimizers = {}
        
        for obj in self.objectives:
            self.networks[obj] = [
                NeuralTSNetwork(dimension, hidden_dim) for _ in range(n_arms)
            ]
            self.optimizers[obj] = [
                optim.Adam(self.networks[obj][a].parameters(), lr=lr)
                for a in range(n_arms)
            ]
        
        # Batch buffers
        self.buffers = {
            obj: {a: {'x': [], 'y': []} for a in range(n_arms)}
            for obj in self.objectives
        }
    
    def predict_all(self, context: np.ndarray) -> List[Dict[str, Tuple[float, float]]]:
        x = torch.FloatTensor(context.flatten())
        predictions = []
        
        for arm in range(self.n_arms):
            arm_pred = {}
            for obj in self.objectives:
                net = self.networks[obj][arm]
                net.eval()
                
                with torch.no_grad():
                    mean = net(x).item()
                
                # Thompson sample via parameter perturbation (CPU, fast)
                original_params = [p.data.clone() for p in net.parameters()]
                with torch.no_grad():
                    for p in net.parameters():
                        p.data += torch.randn_like(p) * self.sigma
                    sampled = net(x).item()
                    for p, orig in zip(net.parameters(), original_params):
                        p.data.copy_(orig)
                
                arm_pred[obj] = (float(mean), float(sampled))
            
            predictions.append(arm_pred)
        
        return predictions
    
    def update(self, context: np.ndarray, arm: int, rewards: Dict[str, float]):
        x = context.flatten()
        
        for obj, reward in rewards.items():
            if obj not in self.buffers:
                continue
            self.buffers[obj][arm]['x'].append(x.copy())
            self.buffers[obj][arm]['y'].append(float(reward))
            
            if len(self.buffers[obj][arm]['x']) >= self.batch_size:
                self._batch_train(obj, arm)
    
    def _batch_train(self, obj: str, arm: int):
        """Batch train on GPU if available."""
        buf = self.buffers[obj][arm]
        if not buf['x']:
            return
        
        X = torch.FloatTensor(np.array(buf['x']))
        Y = torch.FloatTensor(buf['y'])
        
        net = self.networks[obj][arm]
        optimizer = self.optimizers[obj][arm]
        
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
            net.to('cpu')
        
        buf['x'].clear()
        buf['y'].clear()
