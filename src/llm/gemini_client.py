import os
import time
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Optional, Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient:
    """
    Client for interacting with Google's Gemini-3.0 family of models.
    Supports Deep Think, Pro, and Flash variants.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini Client.
        
        Args:
            api_key: Google API Key. If None, reads from GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not found. GeminiClient will not function correctly.")
            raise ValueError("GOOGLE_API_KEY is required.")
            
        genai.configure(api_key=self.api_key)
        
        # Default safety settings - Permissive for simulation context
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

    def get_model(self, model_name: str, generation_config: Optional[Dict] = None) -> genai.GenerativeModel:
        """
        Instantiate a GenerativeModel with specific configuration.
        
        Args:
            model_name: Name of the model (e.g., 'gemini-2.0-flash-exp', 'gemini-1.5-pro').
            generation_config: Configuration dict (temperature, top_p, etc.).
            
        Returns:
            Configured genai.GenerativeModel instance.
        """
        default_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        if generation_config:
            default_config.update(generation_config)
            
        return genai.GenerativeModel(
            model_name=model_name,
            safety_settings=self.safety_settings,
            generation_config=default_config
        )

    def generate_content(self, model_name: str, prompt: str, retries: int = 3) -> Optional[str]:
        """
        Generate content with retry logic.
        
        Args:
            model_name: Model to use.
            prompt: Input text prompt.
            retries: Number of retries on failure.
            
        Returns:
            Generated text or None if failed.
        """
        model = self.get_model(model_name)
        
        for attempt in range(retries):
            try:
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} failed for {model_name}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
        logger.error(f"Failed to generate content with {model_name} after {retries} retries.")
        return None

    def get_embedding(self, text: str, model_name: str = "models/gemini-embedding-001") -> Optional[list[float]]:
        """
        Generate embedding for text.
        
        Args:
            text: Input text.
            model_name: Embedding model name.
            
        Returns:
            List of floats representing the embedding vector.
        """
        try:
            result = genai.embed_content(
                model=model_name,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
