import redis
import numpy as np
from typing import Tuple, Optional
import logging

class RedisClient:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db)
        self.logger = logging.getLogger(__name__)

    def _get_key(self, agent_id: str, suffix: str) -> str:
        return f"agent:{agent_id}:{suffix}"

    def get_model(self, agent_id: str, dim: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieves A_inv and b for a specific agent.
        Returns a tuple (A_inv, b). If keys don't exist, returns (None, None).
        """
        key_A = self._get_key(agent_id, 'A_inv')
        key_b = self._get_key(agent_id, 'b')

        data_A = self.client.get(key_A)
        data_b = self.client.get(key_b)

        if data_A is None or data_b is None:
            return None, None

        # Deserialize bytes to numpy arrays
        # Note: We assume float64 by default as per LinUCB implementation
        # IMPORTANT: Use .copy() to ensure the array is writable, otherwise it's a read-only view of the bytes
        A_inv = np.frombuffer(data_A, dtype=np.float64).reshape((dim, dim)).copy()
        b = np.frombuffer(data_b, dtype=np.float64).reshape((dim, 1)).copy()

        return A_inv, b

    def update_model(self, agent_id: str, A_inv: np.ndarray, b: np.ndarray) -> None:
        """
        Saves A_inv and b to Redis.
        """
        key_A = self._get_key(agent_id, 'A_inv')
        key_b = self._get_key(agent_id, 'b')

        # Serialize numpy arrays to bytes
        self.client.set(key_A, A_inv.tobytes())
        self.client.set(key_b, b.tobytes())
        
    def clear_agent(self, agent_id: str) -> None:
        """Clears data for an agent (useful for testing)"""
        key_A = self._get_key(agent_id, 'A_inv')
        key_b = self._get_key(agent_id, 'b')
        self.client.delete(key_A, key_b)
