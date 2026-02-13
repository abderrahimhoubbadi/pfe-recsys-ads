import numpy as np
from typing import Dict, Tuple

class RewardSimulator:
    """
    Simule la réponse de l'environnement (Clic ou Conversion).
    Supporte maintenant des objectifs multiples : Click (binaire) et Revenue (continu).
    """

    def __init__(self, dimension: int, n_arms: int, seed: int = 42):
        """
        Args:
            dimension (int): Dimension de l'espace de features.
            n_arms (int): Nombre d'actions possibles (publicités).
        """
        self.dimension = dimension
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)
        
        # Hidden parameters for Click (Objective 1)
        self.true_thetas_click = self.rng.standard_normal((n_arms, dimension))
        self.true_thetas_click /= np.linalg.norm(self.true_thetas_click, axis=1, keepdims=True)
        
        # Hidden parameters for Revenue (Objective 2)
        self.true_thetas_revenue = self.rng.standard_normal((n_arms, dimension))
        self.true_thetas_revenue /= np.linalg.norm(self.true_thetas_revenue, axis=1, keepdims=True)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def get_expected_reward(self, context: np.ndarray, arm_index: int) -> Dict[str, float]:
        """
        Calcule les espérances de récompense pour un contexte et un bras donnés.
        Returns:
            Dict with 'click' (probability) and 'revenue' (expected value).
        """
        click_prob = float(self._sigmoid(np.dot(context, self.true_thetas_click[arm_index])))
        revenue_exp = float(self._sigmoid(np.dot(context, self.true_thetas_revenue[arm_index])))
        return {'click': click_prob, 'revenue': revenue_exp}

    def get_reward(self, context: np.ndarray, arm_index: int) -> Dict[str, float]:
        """
        Génère des récompenses échantillonnées.
        Returns:
            Dict with 'click' (0 or 1) and 'revenue' (continuous [0, 1]).
        """
        expected = self.get_expected_reward(context, arm_index)
        click = int(self.rng.random() < expected['click'])
        # Revenue is sampled as: expected * noise (e.g., Beta or scaled uniform)
        revenue = expected['revenue'] * (0.5 + 0.5 * self.rng.random()) if click else 0.0
        return {'click': click, 'revenue': float(revenue)}

    def get_optimal_arm(self, context: np.ndarray, objective: str = 'click') -> int:
        """
        Retourne l'indice du meilleur bras pour un objectif spécifié (Oracle).
        """
        if objective == 'click':
            scores = self.true_thetas_click @ context
        elif objective == 'revenue':
            scores = self.true_thetas_revenue @ context
        else:
            raise ValueError(f"Unknown objective: {objective}")
        return int(np.argmax(scores))

