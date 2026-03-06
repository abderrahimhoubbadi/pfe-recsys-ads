"""
Redis State Store — Persists agent model state to Redis.

Works identically for local Redis (Docker) and GCP Memorystore —
only the REDIS_HOST environment variable changes.
"""

import redis
import numpy as np
import torch
import io
import logging
from typing import Dict, Optional

from src.infra.factory import StateStore

logger = logging.getLogger(__name__)


class RedisStateStore(StateStore):
    """
    Persists agent model state (neural network weights, embeddings, buffers)
    to Redis using binary serialization.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db)
        logger.info(f"RedisStateStore connected to {host}:{port}")

    def _key(self, agent_id: str, field: str) -> str:
        return f"agent:{agent_id}:{field}"

    # ── Generic state save/load ──

    def save_model(self, agent_id: str, state: Dict[str, bytes]) -> None:
        """Save model state as a dict of field → bytes."""
        pipe = self.client.pipeline()
        for field, data in state.items():
            pipe.set(self._key(agent_id, field), data)
        pipe.execute()
        logger.debug(f"Saved state for agent '{agent_id}' ({len(state)} fields)")

    def load_model(self, agent_id: str) -> Optional[Dict[str, bytes]]:
        """Load model state. Returns None if agent has no saved state."""
        # Check if agent exists by looking for a known key
        test_key = self._key(agent_id, "meta")
        if not self.client.exists(test_key):
            return None
        # Retrieve all keys for this agent
        pattern = self._key(agent_id, "*")
        keys = [k.decode() for k in self.client.keys(pattern)]
        if not keys:
            return None
        state = {}
        for key in keys:
            field = key.split(":")[-1]
            state[field] = self.client.get(key)
        return state

    def clear(self, agent_id: str) -> None:
        """Remove all state for an agent."""
        pattern = self._key(agent_id, "*")
        keys = self.client.keys(pattern)
        if keys:
            self.client.delete(*keys)

    # ── PyTorch-specific helpers ──

    def save_torch_model(self, agent_id: str, model_state: dict) -> None:
        """Save a PyTorch state_dict to Redis."""
        buffer = io.BytesIO()
        torch.save(model_state, buffer)
        self.client.set(self._key(agent_id, "torch_state"), buffer.getvalue())
        logger.debug(f"Saved PyTorch state for '{agent_id}'")

    def load_torch_model(self, agent_id: str) -> Optional[dict]:
        """Load a PyTorch state_dict from Redis."""
        data = self.client.get(self._key(agent_id, "torch_state"))
        if data is None:
            return None
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=False)

    # ── Numpy-specific helpers ──

    def save_numpy(self, agent_id: str, field: str, arr: np.ndarray) -> None:
        """Save a numpy array to Redis."""
        buffer = io.BytesIO()
        np.save(buffer, arr)
        self.client.set(self._key(agent_id, field), buffer.getvalue())

    def load_numpy(self, agent_id: str, field: str) -> Optional[np.ndarray]:
        """Load a numpy array from Redis."""
        data = self.client.get(self._key(agent_id, field))
        if data is None:
            return None
        buffer = io.BytesIO(data)
        return np.load(buffer)

    # ── Metadata ──

    def save_meta(self, agent_id: str, meta: dict) -> None:
        """Save agent metadata (JSON-serializable dict)."""
        import json

        self.client.set(self._key(agent_id, "meta"), json.dumps(meta))

    def load_meta(self, agent_id: str) -> Optional[dict]:
        """Load agent metadata."""
        import json

        data = self.client.get(self._key(agent_id, "meta"))
        if data is None:
            return None
        return json.loads(data)

    def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False
