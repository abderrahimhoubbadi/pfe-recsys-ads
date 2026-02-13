import numpy as np

def sherman_morrison_update(A_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Met à jour l'inverse de la matrice de covariance A^{-1} en utilisant la formule de Sherman-Morrison.
    Cette fonction effectue une mise à jour de rang 1 : (A + x.x^T)^{-1}.
    
    Complexité: O(d^2) au lieu de O(d^3) pour une inversion complète.

    Args:
        A_inv (np.ndarray): L'inverse actuel de la matrix A (d, d).
        x (np.ndarray): Le vecteur de contexte (d,) ou (d, 1).

    Returns:
        np.ndarray: La nouvelle matrice inverse mise à jour.
    """
    # Assurer que x est un vecteur colonne (d, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    # Calcul du numérateur : A^{-1} x x^T A^{-1}
    # On commence par v = A^{-1} x qui est utilisé deux fois (O(d^2))
    v = A_inv @ x  # (d, 1)
    
    # Calcul du dénominateur : 1 + x^T A^{-1} x = 1 + x^T v
    # x.T @ v est un scalaire
    denominator = 1.0 + (x.T @ v).item()
    
    # Numérateur : v @ v.T donne une matrice (d, d)  (A^{-1}x)(x^T A^{-1})
    numerator = v @ v.T
    
    # Mise à jour
    A_inv_new = A_inv - (numerator / denominator)
    
    return A_inv_new
