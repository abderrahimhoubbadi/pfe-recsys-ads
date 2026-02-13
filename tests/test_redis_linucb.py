import pytest
import numpy as np
import redis
from src.agents.redis_linucb_agent import RedisLinUCBAgent
from src.infra.redis_client import RedisClient

# Check Redis availability
try:
    r = redis.Redis(host='localhost', port=6379)
    r.ping()
    redis_available = True
except:
    redis_available = False

@pytest.mark.skipif(not redis_available, reason="Redis offline")
class TestRedisLinUCB:
    def setup_method(self):
        self.redis_client = RedisClient()
        self.agent_id = "test_redis_agent"
        self.dim = 3
        self.n_arms = 2
        
        # Clean up
        self._clear_redis()
        
        self.agent = RedisLinUCBAgent(
            agent_id=self.agent_id,
            n_arms=self.n_arms,
            dimension=self.dim,
            redis_client=self.redis_client
        )

    def teardown_method(self):
        self._clear_redis()

    def _clear_redis(self):
        # We need to clear all keys for this agent
        # RedisClient.clear_agent only clears top-level keys in previous implementation?
        # Let's manually clear based on the naming convention in RedisLinUCBAgent
        for arm in range(self.n_arms):
            key_id = f"{self.agent_id}:arm:{arm}"
            self.redis_client.clear_agent(key_id)

    def test_initialization(self):
        """Test that agent initializes and creates keys in Redis on demand"""
        ctx = np.random.rand(self.dim)
        arm = self.agent.select_arm(ctx)
        assert arm in [0, 1]
        
        # Check if keys exist now
        A_inv, b = self.redis_client.get_model(f"{self.agent_id}:arm:{arm}", self.dim)
        assert A_inv is not None
        assert b is not None

    def test_update_flow(self):
        """Test that update changes the model in Redis"""
        ctx = np.random.rand(self.dim)
        arm = 0
        reward = 1.0
        
        # Initial state
        self.agent.select_arm(ctx) # Trigger init
        A_inv_pre, b_pre = self.redis_client.get_model(f"{self.agent_id}:arm:{arm}", self.dim)
        
        # Update
        self.agent.update(ctx, arm, reward)
        
        # Post state
        A_inv_post, b_post = self.redis_client.get_model(f"{self.agent_id}:arm:{arm}", self.dim)
        
        assert not np.array_equal(A_inv_pre, A_inv_post)
        assert not np.array_equal(b_pre, b_post)

    def test_persistence(self):
        """Test that a new agent instance picks up the old state"""
        ctx = np.random.rand(self.dim)
        
        # Train agent 1
        self.agent.update(ctx, 0, 1.0)
        
        # Create agent 2 (same ID)
        agent2 = RedisLinUCBAgent(
            agent_id=self.agent_id,
            n_arms=self.n_arms,
            dimension=self.dim,
            redis_client=self.redis_client
        )
        
        # Verify state matches
        A_inv_1, b_1 = self.agent._fetch_or_init_model(0)
        A_inv_2, b_2 = agent2._fetch_or_init_model(0)
        
        np.testing.assert_array_equal(A_inv_1, A_inv_2)
        np.testing.assert_array_equal(b_1, b_2)
