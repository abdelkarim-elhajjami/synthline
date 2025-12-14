"""
Client for OpenAI and OpenRouter APIs.
"""
import asyncio
from typing import Any, Dict, List, Optional
from openai import AsyncClient
from utils.logger import Logger

class LLMClient:
    """Client for OpenAI and OpenRouter APIs."""
    
    def __init__(self, 
                 logger: Logger,
                 openai_key: Optional[str] = None,
                 openrouter_key: Optional[str] = None,
                 ollama_base_url: Optional[str] = None):
        """Initialize the LLM client with API keys."""
        self._default_openai_key = openai_key
        self._default_openrouter_key = openrouter_key
        self._ollama_base_url = ollama_base_url
        
        self._default_openai_client = None
        self._default_openrouter_client = None
        self._ollama_client = None
        
        self._logger = logger
        self._request_timeout = 120
        self._max_retries = 3

    def _get_client(self, model: str, api_keys: Optional[Dict[str, str]] = None) -> Any:
        """Return the API client for the specified model and keys."""
        keys = api_keys or {}
        
        # 1. Ollama
        if model.startswith('ollama/'):
            if not self._ollama_client:
                self._ollama_client = self._create_async_client(
                    base_url=self._ollama_base_url,
                    api_key="dummy"
                )
            return self._ollama_client
            
        # 2. OpenRouter
        elif model.startswith('openrouter/'):
            key = keys.get('openrouter') or self._default_openrouter_key or keys.get('openai')
            
            if not key or key == self._default_openrouter_key:
                if not self._default_openrouter_client:
                    self._default_openrouter_client = self._create_async_client(
                        api_key=self._default_openrouter_key or "sk-or-v1-dummy",
                        base_url="https://openrouter.ai/api/v1"
                    )
                return self._default_openrouter_client

            return self._create_async_client(
                api_key=key,
                base_url="https://openrouter.ai/api/v1"
            )

        # 3. OpenAI
        else:
            key = keys.get('openai') or self._default_openai_key

            if key == self._default_openai_key:
                 if not self._default_openai_client:
                     self._default_openai_client = self._create_async_client(
                        api_key=self._default_openai_key or "missing-key"
                     )
                 return self._default_openai_client
            
            return self._create_async_client(api_key=key or "missing-key")

    def _create_async_client(self, api_key: str, base_url: Optional[str] = None) -> AsyncClient:
        """Helper to create an AsyncClient instance."""
        return AsyncClient(
            api_key=api_key,
            base_url=base_url,
            timeout=self._request_timeout,
            max_retries=self._max_retries
        )

    async def get_completion(self, 
                             prompt: str,
                             model: str,
                             temperature: float,
                             top_p: float,
                             api_keys: Optional[Dict[str, str]] = None) -> str:
        """Generate a completion for a given prompt using the specified LLM."""
        
        client = self._get_client(model, api_keys)
        
        if model.startswith('ollama/'):
            model_name = model.split('/')[-1]
        elif model.startswith('openrouter/'):
            model_name = model.split('/', 1)[1]
        elif model.startswith('openai/'):
            model_name = model.split('/', 1)[1]
        else:
            model_name = model

        try:
            completion = await client.chat.completions.create(
                model=model_name,
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
                                    features: Dict[str, Any],
                                    api_keys: Optional[Dict[str, str]] = None) -> List[str]:
        """Generate completions for a batch of prompts using the specified LLM."""
        try:
            model = features['llm']
            temperature = float(features['temperature'])
            top_p = float(features['top_p'])
            
            async def _try_completion(prompt: str) -> str:
                try:
                    return await self.get_completion(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        api_keys=api_keys
                    )
                except Exception as e:
                    self._logger.log_error(
                        str(e), 
                        "llm_batch", 
                        {"model": model}
                    )
                    return f"[ERROR: {str(e)[:100]}...]"

            return await asyncio.gather(*(_try_completion(p) for p in prompts))
            
        except Exception as e:
            self._logger.log_error(
                str(e), 
                "llm_batch", 
                {"prompts": [p[:100] + "..." for p in prompts], "model": features['llm'] if 'llm' in features else 'unknown'}
            )
            raise RuntimeError(f"LLM generation failed: {str(e)}") 