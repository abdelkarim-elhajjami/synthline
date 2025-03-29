"""
Client for interacting with LLM APIs.
Supports OpenAI and DeepSeek APIs through a unified interface.
"""
from typing import Any, Dict, List, Optional
from openai import AsyncClient
from utils.logger import Logger

class LLMClient:
    """Client for OpenAI and DeepSeek APIs."""
    
    def __init__(self, 
                 logger: Logger,
                 deepseek_key: Optional[str] = None, 
                 openai_key: Optional[str] = None):
        """Initialize the LLM client with API keys."""
        self._deepseek_key = deepseek_key
        self._openai_key = openai_key
        self._deepseek_client = None
        self._openai_client = None
        self._logger = logger
        
        self._request_timeout = 120
        self._max_retries = 3

    def _get_client(self, model: str) -> AsyncClient:
        """Return the API client for the specified model."""
        
        if model == 'deepseek-chat':
            if not self._deepseek_key:
                error_msg = "DeepSeek API key is missing. Cannot use deepseek-chat model."
                self._logger.log_error(error_msg, "llm", {"model": model})
                raise ValueError(error_msg)
                
            if not self._deepseek_client:
                self._deepseek_client = AsyncClient(
                    api_key=self._deepseek_key,
                    base_url="https://api.deepseek.com",
                    timeout=self._request_timeout,
                    max_retries=self._max_retries
                )
            return self._deepseek_client
            
        elif model == 'gpt-4o':
            if not self._openai_key:
                error_msg = "OpenAI API key is missing. Cannot use gpt-4o model."
                self._logger.log_error(error_msg, "llm", {"model": model})
                raise ValueError(error_msg)
                
            if not self._openai_client:
                self._openai_client = AsyncClient(
                    api_key=self._openai_key,
                    timeout=self._request_timeout,
                    max_retries=self._max_retries
                )
            return self._openai_client
            
        else:
            raise ValueError(f"Unsupported model: {model}")

    async def get_completion(self, 
                             prompt: str,
                             model: str,
                             temperature: float,
                             top_p: float) -> str:
        """Generate a completion for a given prompt using the specified LLM."""
        
        client = self._get_client(model)
        
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p
            )
            
            completion_text = completion.choices[0].message.content
            
            self._logger.log_conversation(
                prompt=prompt,
                completion=completion_text,
                model=model,
                temperature=temperature,
                top_p=top_p
            )
                
            return completion_text
            
        except Exception as e:
            error_message = str(e)            
            self._logger.log_error(
                error_message, 
                "llm", 
                {"prompt": prompt, "model": model}
            )
            raise

    async def get_batch_completions(self,
                                    prompts: List[str],
                                    features: Dict[str, Any]) -> List[str]:
        """Generate completions for a batch of prompts using the specified LLM."""
        
        try:
            model = features['llm']
            temperature = float(features['temperature'])
            top_p = float(features['top_p'])
            
            results = []
            for prompt in prompts:
                try:
                    result = await self.get_completion(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        top_p=top_p
                    )
                    results.append(result)
                
                except Exception as e:
                    results.append(f"[ERROR: {str(e)[:100]}...]")
                    
                    self._logger.log_error(
                        str(e), 
                        "llm_batch", 
                        {"model": model}
                    )
                        
            return results
            
        except Exception as e:
            self._logger.log_error(
                str(e), 
                "llm_batch", 
                {"prompts": [p[:100] + "..." for p in prompts], "model": features['llm'] if 'llm' in features else 'unknown'}
            )
            raise RuntimeError(f"LLM generation failed: {str(e)}") 