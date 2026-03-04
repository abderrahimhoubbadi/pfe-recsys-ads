"""
SentenceTransformer Client — Ultra-fast local embeddings (<10ms).

Replaces the slow Ollama/Llama3.1 embedding pipeline (>200ms)
with a dedicated, lightweight sentence embedding model.
"""

import numpy as np
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformerClient:
    """
    Local embedding client using SentenceTransformers.
    Default model: all-MiniLM-L6-v2 (384-dim, <10ms on CPU).
    """

    _instances: dict = {}  # Singleton cache per model name

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        """Singleton pattern: only load the model once per name."""
        if model_name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[model_name] = instance
        return cls._instances[model_name]

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.model_name = model_name
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dim = {self.embedding_dim}")

    def get_embedding(self, text: str) -> np.ndarray:
        """Encode a single text string into a dense vector."""
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts (more efficient than one-by-one)."""
        return self.model.encode(
            texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False
        )

    def get_dimension(self) -> int:
        return self.embedding_dim
