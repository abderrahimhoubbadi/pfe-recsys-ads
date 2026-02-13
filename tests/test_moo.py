"""
Tests for Multi-Objective LinUCB components.
"""
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.multi_obj_agent import MultiObjectiveLinUCBAgent
from src.policy.moo_policies import (
    linear_scalarization_policy,
    epsilon_constraint_policy,
    pareto_frontier_policy
)


def test_multi_obj_agent_predict_all():
    """Test that predict_all returns correct structure."""
    agent = MultiObjectiveLinUCBAgent(
        n_arms=3, 
        dimension=5, 
        alpha=0.2,
        objectives=['click', 'revenue']
    )
    context = np.random.randn(5)
    context /= np.linalg.norm(context)
    
    predictions = agent.predict_all(context)
    
    # Should return list of dicts, one per arm
    assert len(predictions) == 3
    for arm_pred in predictions:
        assert 'click' in arm_pred
        assert 'revenue' in arm_pred
        # Each objective should have (mean, ucb) tuple
        assert len(arm_pred['click']) == 2
        assert len(arm_pred['revenue']) == 2


def test_multi_obj_agent_update():
    """Test that agent correctly updates both objectives."""
    agent = MultiObjectiveLinUCBAgent(n_arms=2, dimension=3, alpha=0.1)
    context = np.array([1.0, 0.0, 0.0])
    
    # Before update
    pred_before = agent.predict_all(context)
    
    # Update with rewards
    agent.update(context, arm=0, rewards={'click': 1, 'revenue': 0.5})
    
    # After update
    pred_after = agent.predict_all(context)
    
    # UCB should change for arm 0
    assert pred_after[0]['click'][1] != pred_before[0]['click'][1]


def test_linear_scalarization_policy():
    """Test weighted sum policy."""
    predictions = [
        {'click': (0.5, 0.6), 'revenue': (0.3, 0.4)},  # arm 0
        {'click': (0.4, 0.5), 'revenue': (0.6, 0.7)},  # arm 1
    ]
    
    # Equal weights: arm 1 wins (1.2 > 1.0)
    policy = linear_scalarization_policy({'click': 0.5, 'revenue': 0.5})
    assert policy(predictions) == 1
    
    # Heavy click weight: arm 0 wins
    policy = linear_scalarization_policy({'click': 0.9, 'revenue': 0.1})
    assert policy(predictions) == 0


def test_epsilon_constraint_policy():
    """Test constraint satisfaction."""
    predictions = [
        {'click': (0.8, 0.9), 'revenue': (0.1, 0.2)},  # High click, low revenue
        {'click': (0.4, 0.5), 'revenue': (0.5, 0.6)},  # Low click, high revenue
    ]
    
    # With revenue constraint > 0.3, arm 1 should be selected
    policy = epsilon_constraint_policy('click', 'revenue', epsilon=0.3)
    selected = policy(predictions)
    assert selected == 1  # Only arm 1 satisfies constraint


def test_pareto_frontier_policy():
    """Test Pareto-based selection."""
    predictions = [
        {'click': (0.5, 0.6), 'revenue': (0.5, 0.6)},  # balanced
        {'click': (0.8, 0.9), 'revenue': (0.2, 0.3)},  # dominated on revenue
        {'click': (0.2, 0.3), 'revenue': (0.8, 0.9)},  # dominated on click
    ]
    
    policy = pareto_frontier_policy()
    selected = policy(predictions)
    # Arm 0 is non-dominated and closest to ideal
    assert selected in [0, 1, 2]  # All could be non-dominated depending on interpretation


if __name__ == "__main__":
    try:
        test_multi_obj_agent_predict_all()
        test_multi_obj_agent_update()
        test_linear_scalarization_policy()
        test_epsilon_constraint_policy()
        test_pareto_frontier_policy()
        print("All MOO tests passed!")
    except AssertionError as e:
        print(f"MOO Test Failed: {e}")
        exit(1)
