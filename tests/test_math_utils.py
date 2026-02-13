import numpy as np
import pytest
import sys
import os

# Ajout du dossier root au path pour importer src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.math_utils import sherman_morrison_update

def test_sherman_morrison_basic():
    """Vérifie que Sherman-Morrison donne le même résultat que l'inversion numpy standard."""
    d = 5
    # Initialisation A = I
    A = np.eye(d)
    A_inv = np.eye(d)
    
    # Vecteur aléatoire x
    np.random.seed(42)
    x = np.random.randn(d)
    
    # Mise à jour théorique : A_new = A + x x^T
    A_new = A + np.outer(x, x)
    expected_inv = np.linalg.inv(A_new)
    
    # Mise à jour Sherman-Morrison
    calculated_inv = sherman_morrison_update(A_inv, x)
    
    # Vérification
    np.testing.assert_allclose(calculated_inv, expected_inv, rtol=1e-5, atol=1e-8)

def test_sherman_morrison_sequential():
    """Vérifie une séquence de mises à jour."""
    d = 3
    A = np.eye(d)
    A_inv = np.eye(d)
    
    np.random.seed(123)
    
    for _ in range(10):
        x = np.random.randn(d)
        
        # Reference update
        A = A + np.outer(x, x)
        expected_inv = np.linalg.inv(A)
        
        # SM update
        A_inv = sherman_morrison_update(A_inv, x)
        
        np.testing.assert_allclose(A_inv, expected_inv, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    # Permet d'exécuter le test directement sans pytest si besoin
    try:
        test_sherman_morrison_basic()
        test_sherman_morrison_sequential()
        print("Tous les tests Sherman-Morrison ont réussi !")
    except AssertionError as e:
        print(f"Échec du test : {e}")
        exit(1)
    except Exception as e:
        print(f"Erreur : {e}")
        exit(1)
