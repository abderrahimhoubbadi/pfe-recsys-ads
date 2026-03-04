import numpy as np
import logging
from typing import Dict, Any, List, Optional
import functools

from src.agents.llm_agents.base_llm_agent import BaseLLMAgent
from src.agents.multi_obj_agent import MultiObjectiveLinUCBAgent

logger = logging.getLogger(__name__)


class HybridEmbeddingLinUCB(BaseLLMAgent):
    """
    Hybrid Agent: Uses LLM Embeddings to transform context,
    then applies LinUCB on the high-dimensional embedding.
    """

    def __init__(
        self,
        n_arms: int,
        dimension: int,
        embedding_model: str = "models/gemini-embedding-001",
        embedding_dim: int = 768,
    ):

        # Initialize base client
        super().__init__(n_arms, dimension, model_name=embedding_model)
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

        # Internal LinUCB agent working on the text embeddings
        self.internal_agent = MultiObjectiveLinUCBAgent(
            n_arms=n_arms, dimension=embedding_dim, alpha=0.1
        )

        # We don't use the simple dict anymore, use LRU Cache decorator below
        self.api_call_count = 0
        self.cache_hit_count = 0

    @functools.lru_cache(maxsize=1000)
    def _fetch_embedding_cached(self, rounded_context_tuple: tuple) -> tuple:
        """
        Cached method to fetch embeddings.
        Tuple is used because lists/arrays are not hashable for lru_cache.
        """
        text_context = f"User feature vector: {list(rounded_context_tuple)}"
        emb_list = self.client.get_embedding(text_context, self.embedding_model)

        if emb_list is None:
            # Fallback
            return tuple(np.random.normal(0, 0.1, self.embedding_dim).tolist())

        return tuple(emb_list)

    def _get_embedding(self, context: np.ndarray) -> np.ndarray:
        """
        Get embedding for the context string with optimized caching.
        """
        # Round the continuous variables to 1 decimal to massively increase cache hits
        # meaning a user with [0.12] and [0.14] are treated semantically identical by the LLM
        rounded_context = np.round(context, decimals=1)
        context_tuple = tuple(rounded_context.tolist())

        # For statistics
        cache_info = self._fetch_embedding_cached.cache_info()
        hits_before = cache_info.hits

        # Call cached function
        emb_tuple = self._fetch_embedding_cached(context_tuple)

        hits_after = self._fetch_embedding_cached.cache_info().hits
        if hits_after > hits_before:
            self.cache_hit_count += 1
        else:
            self.api_call_count += 1

        emb_vec = np.array(emb_tuple)

        # Verify dimension
        if emb_vec.shape[0] != self.embedding_dim:
            if emb_vec.shape[0] > self.embedding_dim:
                emb_vec = emb_vec[: self.embedding_dim]
            else:
                emb_vec = np.pad(emb_vec, (0, self.embedding_dim - emb_vec.shape[0]))

        return emb_vec

    def select_arm(self, context: np.ndarray, policy=None) -> int:
        emb_context = self._get_embedding(context)
        return self.internal_agent.select_arm(emb_context, policy=policy)

    def update(self, context: np.ndarray, arm: int, rewards: Dict[str, float]):
        emb_context = self._get_embedding(context)
        self.internal_agent.update(emb_context, arm, rewards)

    def get_model_params(self) -> Dict[str, Any]:
        params = super().get_model_params()
        params.update(
            {
                "embedding_model": self.embedding_model,
                "embedding_dim": self.embedding_dim,
                "api_calls": self.api_call_count,
                "cache_hits": self.cache_hit_count,
                "internal_linucb": self.internal_agent.get_model_params(),
            }
        )
        return params
