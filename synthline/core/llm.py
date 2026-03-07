"""
Client for OpenAI, OpenRouter, and HuggingFace APIs.
"""
import asyncio
from typing import Any, Dict, List, Optional
from openai import AsyncClient
from huggingface_hub import AsyncInferenceClient
from synthline.utils.logger import Logger

class LLMClient:
    """Client for OpenAI, OpenRouter, and HuggingFace APIs."""
    def __init__(self, 
                 logger: Logger,
                 openai_key: Optional[str] = None,
                 openrouter_key: Optional[str] = None,
                 ollama_base_url: Optional[str] = None,
                 hf_token: Optional[str] = None):
        """Initialize the LLM client with API keys."""
        self._default_openai_key = openai_key
        self._default_openrouter_key = openrouter_key
        self._ollama_base_url = ollama_base_url
        self._hf_token = hf_token
        
        self._default_openai_client = None
        self._default_openrouter_client = None
        self._ollama_client = None
        self._hf_client = None
        
        self._logger = logger
        self._request_timeout = 120
        self._max_retries = 3

    @staticmethod
    def _provider_for_model(model: str) -> str:
        if model.startswith("ollama/"):
            return "ollama"
        if model.startswith("huggingface/"):
            return "huggingface"
        if model.startswith("openrouter/"):
            return "openrouter"
        if model.startswith("openai/"):
            return "openai"
        return "openai"

    @staticmethod
    def _model_name_for_request(model: str) -> str:
        provider = LLMClient._provider_for_model(model)
        if provider == "ollama":
            return model.split("/")[-1]
        if "/" in model and provider in {"huggingface", "openrouter", "openai"}:
            return model.split("/", 1)[1]
        return model

    def _get_client(self, model: str, api_keys: Optional[Dict[str, str]] = None) -> Any:
        """Return the API client for the specified model and keys."""
        keys = api_keys or {}
        provider = self._provider_for_model(model)
        
        # 1. Ollama
        if provider == "ollama":
            if not self._ollama_client:
                self._ollama_client = self._create_async_client(
                    base_url=self._ollama_base_url,
                    api_key="dummy"
                )
            return self._ollama_client
        
        # 2. HuggingFace
        elif provider == "huggingface":
            if not self._hf_client:
                self._hf_client = AsyncInferenceClient(token=self._hf_token)
            return self._hf_client
            
        # 3. OpenRouter
        elif provider == "openrouter":
            key = keys.get('openrouter') or self._default_openrouter_key
            
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

        # 4. OpenAI
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
        provider = self._provider_for_model(model)
        model_name = self._model_name_for_request(model)

        try:
            if provider == "huggingface":
                response = await client.chat_completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=4096
                )
                completion_text = response.choices[0].message.content
            else:
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
            self._logger.log_error(
                str(e),
                "llm",
                {"prompt": prompt, "model": model},
            )
            raise

    async def get_batch_completions(self,
                                    prompts: List[str],
                                    features: Dict[str, Any],
                                    api_keys: Optional[Dict[str, str]] = None) -> List[str]:
        """Generate completions for a batch of prompts using the specified LLM."""
        tasks: List[asyncio.Task] = []
        try:
            model = features['llm']
            temperature = float(features['temperature'])
            top_p = float(features['top_p'])

            async def _completion_task(prompt_idx: int, prompt_text: str) -> tuple[int, str]:
                completion_text = await self.get_completion(
                    prompt=prompt_text,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    api_keys=api_keys,
                )
                return prompt_idx, completion_text

            tasks = [
                asyncio.create_task(_completion_task(idx, prompt))
                for idx, prompt in enumerate(prompts)
            ]
            completions: List[str] = [""] * len(prompts)

            for task in asyncio.as_completed(tasks):
                idx, completion = await task
                completions[idx] = completion

            return completions

        except Exception as e:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            self._logger.log_error(
                str(e),
                "llm_batch",
                {
                    "prompts": [p[:100] + "..." for p in prompts],
                    "model": features['llm'] if 'llm' in features else 'unknown',
                }
            )
            raise
            