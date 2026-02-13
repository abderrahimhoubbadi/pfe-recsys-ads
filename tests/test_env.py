import numpy as np
import pytest
import sys
import os

# Ajout du dossier root au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env.context_generator import ContextGenerator
from src.env.reward_simulator import RewardSimulator

def test_context_shape():
    d = 10
    gen = ContextGenerator(dimension=d)
    ctx = gen.get_context()
    
    assert ctx.shape == (d,)
    # Verify normalization
    assert np.isclose(np.linalg.norm(ctx), 1.0)

def test_reward_mechanics():
    """Test multi-objective reward simulator."""
    d = 5
    n_arms = 3
    sim = RewardSimulator(dimension=d, n_arms=n_arms)
    ctx = np.random.randn(d)
    ctx /= np.linalg.norm(ctx)
    
    # Test expected reward returns Dict
    expected = sim.get_expected_reward(ctx, arm_index=0)
    assert isinstance(expected, dict)
    assert 'click' in expected and 'revenue' in expected
    assert 0.0 <= expected['click'] <= 1.0
    assert 0.0 <= expected['revenue'] <= 1.0
    
    # Test sampled reward returns Dict
    reward = sim.get_reward(ctx, arm_index=0)
    assert isinstance(reward, dict)
    assert reward['click'] in [0, 1]
    assert 0.0 <= reward['revenue'] <= 1.0
    
    # Test optimal arm for each objective
    opt_click = sim.get_optimal_arm(ctx, objective='click')
    opt_revenue = sim.get_optimal_arm(ctx, objective='revenue')
    assert 0 <= opt_click < n_arms
    assert 0 <= opt_revenue < n_arms

if __name__ == "__main__":
    try:
        test_context_shape()
        test_reward_mechanics()
        print("Tests Environnement OK !")
    except AssertionError as e:
        print(f"Test Environnement Failed: {e}")
        exit(1)

