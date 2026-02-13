import pytest
import numpy as np
import redis
from src.infra.redis_client import RedisClient

# Check if Redis is available, otherwise skip tests
try:
    r = redis.Redis(host='localhost', port=6379)
    r.ping()
    redis_available = True
except redis.ConnectionError:
    redis_available = False

@pytest.mark.skipif(not redis_available, reason="Redis is not running")
class TestRedisClient:
    def setup_method(self):
        self.client = RedisClient()
        self.agent_id = "test_agent_001"
        self.dim = 5
        # Clean up before test
        self.client.clear_agent(self.agent_id)

    def teardown_method(self):
        # Clean up after test
        self.client.clear_agent(self.agent_id)

    def test_round_trip(self):
        """Test getting and setting model parameters"""
        # Create dummy data
        A_inv = np.random.rand(self.dim, self.dim)
        b = np.random.rand(self.dim, 1)

        # Save to Redis
        self.client.update_model(self.agent_id, A_inv, b)

        # Retrieve from Redis
        retrieved_A, retrieved_b = self.client.get_model(self.agent_id, self.dim)

        # Verify
        assert retrieved_A is not None
        assert retrieved_b is not None
        np.testing.assert_array_equal(A_inv, retrieved_A)
        np.testing.assert_array_equal(b, retrieved_b)

    def test_get_non_existent(self):
        """Test behavior when keys don't exist"""
        A_inv, b = self.client.get_model(self.agent_id, self.dim)
        assert A_inv is None
        assert b is None
