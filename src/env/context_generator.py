import numpy as np

class ContextGenerator:
    """
    Simule des contextes utilisateurs (vecteurs de features).
    Dans la Phase 1, on utilise des vecteurs aléatoires.
    Plus tard, on pourra simuler des distributions plus réalistes (ex: heure de la journée, device).
    """
    
    def __init__(self, dimension: int = 10, seed: int = 42):
        """
        Args:
            dimension (int): Dimension du vecteur de contexte (d).
            seed (int): Graine aléatoire pour la reproductibilité.
        """
        self.dimension = dimension
        self.rng = np.random.default_rng(seed)

    def get_context(self) -> np.ndarray:
        """
        Génère un vecteur de contexte aléatoire normalisé ($||x|| = 1$).
        On normalise souvent pour stabiliser l'apprentissage des Bandits Linéaires.
        
        Returns:
            np.ndarray: Vecteur de taille (d,).
        """
        # Générer un vecteur aléatoire depuis une distribution normale
        raw_context = self.rng.standard_normal(self.dimension)
        
        # Normalisation L2
        norm = np.linalg.norm(raw_context)
        if norm == 0:
            return raw_context # Should practically never happen
            
        return raw_context / norm

    def get_batch(self, batch_size: int) -> np.ndarray:
        """
        Retourne un batch de contextes (utile pour les tests ou simulation batch).
        """
        raw_batch = self.rng.standard_normal((batch_size, self.dimension))
        norms = np.linalg.norm(raw_batch, axis=1, keepdims=True)
        return raw_batch / norms
