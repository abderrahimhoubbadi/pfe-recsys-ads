import json
import numpy as np
import logging
import ast
import re
from typing import Dict, Any

from src.agents.llm_agents.base_llm_agent import BaseLLMAgent

logger = logging.getLogger(__name__)

class GeminiProAgent(BaseLLMAgent):
    """
    Agent utilizing Gemini 3.0 Pro for general instruction following.
    """
    
    def __init__(self, n_arms: int, dimension: int, model_name: str = "models/gemini-flash-latest"):
        super().__init__(n_arms, dimension, model_name=model_name)
    
    def _create_prompt(self, context: np.ndarray) -> str:
        """
        Create a direct instruction prompt.
        """
        prompt = f"""
        Context: User Information Vector {context.tolist()}
        Action Space: {self.n_arms} ads available (Index 0 to {self.n_arms - 1}).
        
        Task: Select the best ad for this user to maximize engagement and revenue.
        
        Do NOT write code. Do NOT output python or javascript.
        Perform the selection yourself and return ONLY a JSON object with the result.
        
        Return ONLY a JSON object with the key "action":
        {{ "action": <integer_index> }}
        """
        return prompt

    def select_arm(self, context: np.ndarray, policy=None) -> int:
        # Policy ignored, LLM is the policy.
        prompt = self._create_prompt(context)
        response_text = self.client.generate_content(self.model_name, prompt)
        
        if not response_text:
            return np.random.randint(self.n_arms)
            
        try:
            # Clean response
            text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Robust extraction
            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                text = text[start:end]
            
            # Strategy 1: Standard JSON
            try:
                data = json.loads(text)
                action = int(data.get("action", 0))
            except Exception:
                # Strategy 2: Python/JS style (single quotes)
                try:
                    data = ast.literal_eval(text)
                    action = int(data.get("action", 0))
                except Exception:
                    # Strategy 3: Regex fallback
                    # Matches: "action": 5, action: 5, Action: 5, etc.
                    match = re.search(r'[\'"]?action[\'"]?\s*[:=]\s*(\d+)', text, re.IGNORECASE)
                    if match:
                        action = int(match.group(1))
                    else:
                        raise ValueError("No action found")
            
            # Bound check
            if action < 0 or action >= self.n_arms:
                action = max(0, min(action, self.n_arms - 1))
                
            return action
        except Exception as e:
            logger.warning(f"ProAgent error: {e}")
            return np.random.randint(self.n_arms)
