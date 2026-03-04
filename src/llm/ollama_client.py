import requests
import json
import logging
import os
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for interacting with a local Ollama instance.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.embeddings_endpoint = f"{self.base_url}/api/embeddings"
        
    def generate_content(self, model: str, prompt: str) -> Optional[str]:
        """
        Generate text using a local Ollama model.
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.generate_endpoint, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data.get("response")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama generation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Ollama generation: {e}")
            return None

    def get_embedding(self, text: str, model_name: str = "llama3.1") -> Optional[List[float]]:
        """
        Generate embeddings using a local Ollama model.
        """
        try:
            payload = {
                "model": model_name,
                "prompt": text
            }
            
            response = requests.post(self.embeddings_endpoint, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data.get("embedding")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama embedding failed: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Ollama embedding: {e}")
            return None
