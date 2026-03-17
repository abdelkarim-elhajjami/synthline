"""
Client for OpenAI, OpenRouter, and HuggingFace APIs.
"""
import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncClient, RateLimitError, APIStatusError, APITimeoutError, APIConnectionError
from huggingface_hub import AsyncInferenceClient
from synthline.utils.logger import Logger


class LLMClient:
    """Client for OpenAI, OpenRouter, and HuggingFace APIs."""

    REQUEST_TIMEOUT = 120
    MAX_RETRIES = 5
    MAX_CONCURRENCY = 100
    DEFAULT_MAX_TOKENS = 32768

    def __init__(self,
                 logger: Logger,
                 openai_key: Optional[str] = None,
                 openrouter_key: Optional[str] = None,
                 ilaas_key: Optional[str] = None,
                 ollama_base_url: Optional[str] = None,
                 hf_token: Optional[str] = None):
        """Initialize the LLM client with API keys."""
        self._default_openai_key = openai_key
        self._default_openrouter_key = openrouter_key
        self._default_ilaas_key = ilaas_key
        self._ollama_base_url = ollama_base_url
        self._hf_token = hf_token

        self._default_openai_client = None
        self._default_openrouter_client = None
        self._default_ilaas_client = None
        self._ollama_client = None
        self._hf_client = None

        self._logger = logger
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENCY)

    @staticmethod
    def _provider_for_model(model: str) -> str:
        if model.startswith("ollama/"):
            return "ollama"
        if model.startswith("huggingface/"):
            return "huggingface"
        if model.startswith("openrouter/"):
            return "openrouter"
        if model.startswith("ilaas/"):
            return "ilaas"
        if model.startswith("openai/"):
            return "openai"
        return "openai"

    @staticmethod
    def _model_name_for_request(model: str) -> str:
        provider = LLMClient._provider_for_model(model)
        if provider == "ollama":
            return model.split("/")[-1]
        if "/" in model and provider in {"huggingface", "openrouter", "ilaas", "openai"}:
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

            if key == self._default_openrouter_key:
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

        # 4. ILaaS
        elif provider == "ilaas":
            key = keys.get('ilaas') or self._default_ilaas_key

            if key == self._default_ilaas_key:
                if not self._default_ilaas_client:
                    self._default_ilaas_client = self._create_async_client(
                        api_key=self._default_ilaas_key or "missing-key",
                        base_url="https://llm.ilaas.fr/v1"
                    )
                return self._default_ilaas_client

            return self._create_async_client(
                api_key=key,
                base_url="https://llm.ilaas.fr/v1"
            )

        # 5. OpenAI
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
            timeout=self.REQUEST_TIMEOUT,
            max_retries=0,  # We handle retries ourselves for proper rate limit adaptation
        )

    @staticmethod
    def _parse_retry_after(error: APIStatusError) -> Optional[float]:
        """Extract retry-after delay from API error response headers."""
        headers = getattr(error, 'response', None)
        if headers is None:
            return None
        headers = getattr(headers, 'headers', None)
        if headers is None:
            return None
        # Standard header
        retry_after = headers.get('retry-after')
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass
        # OpenAI / OpenRouter duration format (e.g. "6m30s", "1s", "200ms")
        reset = headers.get('x-ratelimit-reset-requests')
        if reset:
            return LLMClient._parse_duration(reset)
        return None

    @staticmethod
    def _parse_duration(duration: str) -> Optional[float]:
        """Parse duration string like '6m30s' or '200ms' to seconds."""
        units = {'h': 3600, 'm': 60, 's': 1, 'ms': 0.001}
        total = 0.0
        for value, unit in re.findall(r'(\d+\.?\d*)(ms|[hms])', duration):
            total += float(value) * units[unit]
        return total if total > 0 else None

    async def get_completion(self,
                             prompt: str,
                             model: str,
                             temperature: float,
                             top_p: float,
                             api_keys: Optional[Dict[str, str]] = None,
                             response_format: Optional[Dict[str, Any]] = None,
                             reasoning: Optional[Dict[str, Any]] = None) -> str:
        """Generate a completion for a given prompt using the specified LLM.

        Retries automatically on rate limit (429) and transient server errors
        (500, 502, 503), using retry-after headers when available and
        exponential backoff as fallback.
        """
        client = self._get_client(model, api_keys)
        provider = self._provider_for_model(model)
        model_name = self._model_name_for_request(model)
        last_error: Optional[Exception] = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": self.DEFAULT_MAX_TOKENS,
                }
                if reasoning:
                    kwargs.setdefault("extra_body", {})["reasoning"] = reasoning

                if response_format:
                    if provider == "ollama" and "json_schema" in response_format:
                        kwargs["format"] = response_format["json_schema"]["schema"]
                    elif provider != "ollama":
                        kwargs["response_format"] = response_format

                if provider == "huggingface":
                    response = await client.chat_completion(**kwargs)
                else:
                    response = await client.chat.completions.create(**kwargs)
                completion_text = response.choices[0].message.content

                self._logger.log_conversation(
                    prompt=prompt,
                    completion=completion_text,
                    model=model,
                    temperature=temperature,
                    top_p=top_p
                )

                return completion_text

            except RateLimitError as e:
                last_error = e
                retry_after = self._parse_retry_after(e)
                wait = retry_after if retry_after else min(2 ** attempt, 60)
                self._logger.log_error(
                    f"Rate limit, retry {attempt + 1}/{self.MAX_RETRIES + 1} in {wait:.0f}s",
                    "llm", {"model": model},
                )
                await asyncio.sleep(wait)

            except APIStatusError as e:
                if e.status_code in (500, 502, 503):
                    last_error = e
                    wait = min(2 ** attempt, 60)
                    self._logger.log_error(
                        f"Server {e.status_code}, retry {attempt + 1}/{self.MAX_RETRIES + 1} in {wait:.0f}s",
                        "llm", {"model": model},
                    )
                    await asyncio.sleep(wait)
                else:
                    self._logger.log_error(str(e), "llm", {"model": model})
                    raise

            except (APITimeoutError, APIConnectionError, json.JSONDecodeError) as e:
                last_error = e
                wait = min(2 ** attempt, 60)
                self._logger.log_error(
                    f"{type(e).__name__}, retry {attempt + 1}/{self.MAX_RETRIES + 1} in {wait:.0f}s",
                    "llm", {"model": model},
                )
                await asyncio.sleep(wait)

            except Exception as e:
                self._logger.log_error(
                    str(e),
                    "llm",
                    {"prompt": prompt, "model": model},
                )
                raise

        # All retries exhausted
        self._logger.log_error(
            f"All {self.MAX_RETRIES + 1} attempts failed",
            "llm",
            {"prompt": prompt[:100], "model": model},
        )
        raise last_error  # type: ignore[misc]

    async def get_batch_completions(self,
                                    prompts: List[str],
                                    features: Dict[str, Any],
                                    api_keys: Optional[Dict[str, str]] = None,
                                    response_format: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate completions for a batch of prompts using the specified LLM."""
        tasks: List[asyncio.Task] = []
        try:
            model = features['llm']
            temperature = float(features['temperature'])
            top_p = float(features['top_p'])
            reasoning = features.get('reasoning')

            async def _completion_task(prompt_idx: int, prompt_text: str) -> Tuple[int, str]:
                async with self._semaphore:
                    completion_text = await self.get_completion(
                        prompt=prompt_text,
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        api_keys=api_keys,
                        response_format=response_format,
                        reasoning=reasoning,
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
