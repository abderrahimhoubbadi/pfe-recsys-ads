"""
Deep Contextual Bandits Agent (Bootstrap Ensemble) — Optimized.

CPU inference + GPU batch training per ensemble member.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from .base_moo_agent import BaseMOOAgent

GPU_AVAILABLE = torch.cuda.is_available()
TRAIN_DEVICE = torch.device('cuda' if GPU_AVAILABLE else 'cpu')


class EnsembleMLP(nn.Module):
    """Single MLP member of the ensemble."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DeepBanditAgent(BaseMOOAgent):
    """
    Deep Contextual Bandits with Bootstrap Ensemble — Optimized.
    
    CPU inference (fast per-sample) + GPU batch training.
    Uncertainty = std across ensemble predictions.
    """
    
    def __init__(self, n_arms: int, dimension: int,
                 alpha: float = 1.0, n_ensemble: int = 5,
                 hidden_dim: int = 32, lr: float = 0.01,
                 batch_size: int = 16,
                 objectives: List[str] = None):
        super().__init__(n_arms, dimension, objectives)
        self.alpha = alpha
        self.n_ensemble = n_ensemble
        self.batch_size = batch_size
        
        # Ensembles on CPU for inference
        self.ensembles = {}
        self.optimizers = {}
        
        for obj in self.objectives:
            self.ensembles[obj] = [
                [EnsembleMLP(dimension, hidden_dim) for _ in range(n_ensemble)]
                for _ in range(n_arms)
            ]
            self.optimizers[obj] = [
                [optim.Adam(self.ensembles[obj][a][e].parameters(), lr=lr)
                 for e in range(n_ensemble)]
                for a in range(n_arms)
            ]
        
        # Batch buffers per (arm, objective, ensemble_member)
        self.buffers = {
            obj: {a: {'x': [], 'y': [], 'mask': []} for a in range(n_arms)}
            for obj in self.objectives
        }
    
    def predict_all(self, context: np.ndarray) -> List[Dict[str, Tuple[float, float]]]:
        x = torch.FloatTensor(context.flatten())
        predictions = []
        
        for arm in range(self.n_arms):
            arm_pred = {}
            for obj in self.objectives:
                ensemble = self.ensembles[obj][arm]
                
                # Fast CPU inference across ensemble
                preds = []
                with torch.no_grad():
                    for net in ensemble:
                        net.eval()
                        preds.append(net(x).item())
                
                mean = float(np.mean(preds))
                std = float(np.std(preds))
                arm_pred[obj] = (mean, mean + self.alpha * std)
            
            predictions.append(arm_pred)
        
        return predictions
    
    def update(self, context: np.ndarray, arm: int, rewards: Dict[str, float]):
        x = context.flatten()
        
        # Bootstrap mask: each ensemble member sees this sample with p=0.7
        mask = [1 if np.random.random() < 0.7 else 0 for _ in range(self.n_ensemble)]
        
        for obj, reward in rewards.items():
            if obj not in self.buffers:
                continue
            self.buffers[obj][arm]['x'].append(x.copy())
            self.buffers[obj][arm]['y'].append(float(reward))
            self.buffers[obj][arm]['mask'].append(mask.copy())
            
            if len(self.buffers[obj][arm]['x']) >= self.batch_size:
                self._batch_train(obj, arm)
    
    def _batch_train(self, obj: str, arm: int):
        """Batch train ensemble on GPU if available."""
        buf = self.buffers[obj][arm]
        if not buf['x']:
            return
        
        X = torch.FloatTensor(np.array(buf['x']))
        Y = torch.FloatTensor(buf['y'])
        masks = np.array(buf['mask'])  # (batch, n_ensemble)
        
        for e in range(self.n_ensemble):
            # Select samples this ensemble member should see
            sample_mask = masks[:, e].astype(bool)
            if not np.any(sample_mask):
                continue
            
            X_e = X[sample_mask]
            Y_e = Y[sample_mask]
            
            net = self.ensembles[obj][arm][e]
            optimizer = self.optimizers[obj][arm][e]
            
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
                net.to('cpu')
        
        buf['x'].clear()
        buf['y'].clear()
        buf['mask'].clear()
