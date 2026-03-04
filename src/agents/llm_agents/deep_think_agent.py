import json
import numpy as np
import logging
import ast
import re
from typing import Dict, Any

from src.agents.llm_agents.base_llm_agent import BaseLLMAgent

logger = logging.getLogger(__name__)

class GeminiDeepThinkAgent(BaseLLMAgent):
    """
    Agent utilizing Gemini 3.0's reasoning capabilities (Deep Think).
    """
    
    def __init__(self, n_arms: int, dimension: int, model_name: str = "models/gemini-flash-latest"):
        super().__init__(n_arms, dimension, model_name=model_name)
    
    def _create_prompt(self, context: np.ndarray) -> str:
        """
        Create a prompt asking for reasoning and action selection.
        """
        # In a real scenario, context would be richer.
        # Here we simulate the task: "Given user features X, choose best ad 0..K-1"
        
        prompt = f"""
        You are an advanced recommendation agent. Your goal is to select the best advertisement (arm) for a user based on their context vector.
        
        Context Vector: {context.tolist()}
        Number of available arms (Ads): {self.n_arms}
        
        Your task:
        1. Analyze the context vector values.
        2. Reason about which arm might be best (simulate a reasoning process).
        3. Select an arm index from 0 to {self.n_arms - 1}.
        
        Do NOT write code. Do NOT output python or javascript.
        Perform the selection yourself and return ONLY a JSON object with the result.
        
        Output strictly in JSON format:
        {{
            "reasoning_trace": "string explaining your thought process",
            "final_action": integer (0 to {self.n_arms - 1})
        }}
        """
        return prompt

    def select_arm(self, context: np.ndarray, policy=None) -> int:
        # Policy is ignored here as the LLM decides itself.
        prompt = self._create_prompt(context)
        response_text = self.client.generate_content(self.model_name, prompt)
        
        if not response_text:
            logger.warning("DeepThinkAgent: No response from Gemini. Picking random arm.")
            return np.random.randint(self.n_arms)
            
        try:
            # Clean response (sometimes models output ```json ... ```)
            text = response_text.replace("```json", "").replace("```", "").replace("```javascript", "").strip()
            
            # Robust extraction: find the first { and the last }
            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                text = text[start:end]
                
            # Strategy 1: Standard JSON
            try:
                data = json.loads(text)
                action = int(data.get("final_action", 0))
            except Exception:
                # Strategy 2: Python/JS style (single quotes)
                try:
                    data = ast.literal_eval(text)
                    action = int(data.get("final_action", 0))
                except Exception:
                    # Strategy 3: Regex fallback (find "final_action": <digits>)
                    # Matches: "final_action": 5, Final Action: 5, etc.
                    match = re.search(r'(?:final_action|Final Action|final action)\s*[:=]\s*(\d+)', text, re.IGNORECASE)
                    if match:
                        action = int(match.group(1))
                    else:
                         # Last resort: just find the last number in the text? No, too risky.
                        raise ValueError("No final_action found")
            
            # Bound check
            if action < 0 or action >= self.n_arms:
                logger.warning(f"DeepThinkAgent: Action {action} out of bounds. Clipping.")
                action = max(0, min(action, self.n_arms - 1))
                
            return action
        except Exception as e:
            logger.error(f"DeepThinkAgent: Error processing response: {e}")
            return np.random.randint(self.n_arms)
