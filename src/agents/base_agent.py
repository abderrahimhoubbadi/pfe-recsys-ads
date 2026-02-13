from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BaseAgent(ABC):
    """
    Classe abstraite définissant l'interface d'un agent Bandit.
    """
    
    def __init__(self, n_arms: int, dimension: int):
        self.n_arms = n_arms
        self.dimension = dimension

    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        """
        Choisit une action (bras) basée sur le contexte donné.
        
        Args:
            context (np.ndarray): Le vecteur de contexte utilisateur (d,).
            
        Returns:
            int: L'index du bras choisi.
        """
        pass

    @abstractmethod
    def update(self, context: np.ndarray, arm: int, reward: float):
        """
        Met à jour les paramètres de l'agent après avoir observé une récompense.
        
        Args:
            context (np.ndarray): Le vecteur de contexte utilisé (d,).
            arm (int): L'index du bras qui a été choisi.
            reward (float): La récompense obtenue (0 ou 1, ou continue).
        """
        pass

    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Retourne les paramètres internes (pour debugging/logging)."""
        pass
