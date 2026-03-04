import numpy as np
from typing import Dict, Any, Optional
import json
import logging
from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class BaseLLMAgent(BaseAgent):
    """
    Base class for LLM-based bandit agents.
    Handles Client initialization (Gemini or Ollama) and common utilities.
    """
    
    def __init__(self, n_arms: int, dimension: int, model_name: str):
        super().__init__(n_arms, dimension)
        self.model_name = model_name
        
        # Determine which client to use based on model name pattern
        if model_name.startswith("models/"):
            logger.info(f"Initializing GeminiClient for model: {model_name}")
            self.client = GeminiClient()
        else:
            logger.info(f"Initializing OllamaClient for model: {model_name}")
            self.client = OllamaClient()
            
        self.history = [] # Optional: Store interaction history
        
    def _context_to_text(self, context: np.ndarray) -> str:
        """
        Convert vector context to a text description.
        Since we only have a vector, we provide a generic description.
        In a real app, we would have the raw features (user features, etc.).
        """
        # Simplification: Describe the vector as a list of features
        return f"User Context Vector: {context.tolist()}"

    def select_arm(self, context: np.ndarray, policy=None) -> int:
        """
        Default implementation - should be overridden by subclasses.
        Accepts optional policy for compatibility with global_comparison interface.
        """
        raise NotImplementedError

    def update(self, context: np.ndarray, arm: int, reward: float):
        """
        Default update - just log or store in history.
        LLMs might not 'train' in the traditional sense, but can use history in prompt (In-Context Learning).
        """
        self.history.append({
            "context": context.tolist(),
            "arm": arm,
            "reward": reward
        })
        # Keep history manageable
        if len(self.history) > 20: 
            self.history.pop(0)

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "history_len": len(self.history)
        }
