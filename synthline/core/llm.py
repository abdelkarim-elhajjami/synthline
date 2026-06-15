"""
Multi-provider LLM client.
"""
import asyncio
import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncClient, RateLimitError, APIStatusError, APITimeoutError, APIConnectionError
from huggingface_hub import AsyncInferenceClient
from synthline.core.model_compatibility import validate_model_compatibility
from synthline.errors import (
    ProviderConfigurationError,
    StructuredOutputCompatibilityError,
    StructuredOutputError,
    structured_output_response_error,
)
from synthline.utils.logger import Logger

_PROVIDER_PREFIXES = {
    "ollama/": "ollama",
    "huggingface/": "huggingface",
    "openrouter/": "openrouter",
    "ilaas/": "ilaas",
    "openai/": "openai",
}

def _response_format_for_model(
    model_name: str,
    response_format: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize the shared schema only where a model family requires it."""
    if not model_name.startswith("anthropic/"):
        return response_format

    # Claude structured outputs reject some JSON Schema constraints. Synthline
    # still validates the exact sample count after generation.
    rf = copy.deepcopy(response_format)
    schema = rf.get("json_schema", {}).get("schema", {})
    for prop in schema.get("properties", {}).values():
        if prop.get("type") == "array":
            prop.pop("minItems", None)
            prop.pop("maxItems", None)
    return rf


class LLMClient:
    """Client for OpenAI-compatible APIs and Hugging Face inference."""

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
        for prefix, provider in _PROVIDER_PREFIXES.items():
            if model.startswith(prefix):
                return provider
        return "openai"

    @staticmethod
    def _model_name_for_request(model: str) -> str:
        for prefix in _PROVIDER_PREFIXES:
            if model.startswith(prefix):
                return model[len(prefix):]
        return model

    @staticmethod
    def _normalize_ollama_base_url(base_url: Optional[str]) -> str:
        """Return an Ollama OpenAI-compatible base URL."""
        if not base_url or not base_url.strip():
            raise ProviderConfigurationError(
                "Ollama is not configured. Set OLLAMA_BASE_URL to the Ollama server URL "
                "(for example, http://localhost:11434) and try again."
            )
        normalized = base_url.strip().rstrip("/")
        return normalized if normalized.endswith("/v1") else f"{normalized}/v1"

    def _get_client(self, model: str, api_keys: Optional[Dict[str, str]] = None) -> Any:
        """Return the API client for the specified model and keys."""
        keys = api_keys or {}
        provider = self._provider_for_model(model)

        # 1. Ollama
        if provider == "ollama":
            if not self._ollama_client:
                self._ollama_client = self._create_async_client(
                    base_url=self._normalize_ollama_base_url(self._ollama_base_url),
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
    def _is_structured_output_compatibility_error(error: Exception) -> bool:
        """Return whether an error indicates unsupported structured-output parameters."""
        details = " ".join([
            str(error),
            str(getattr(error, "body", "")),
            str(getattr(error, "message", "")),
        ]).lower()
        structured_terms = (
            "structured output",
            "structured_outputs",
            "json_schema",
            "response_format",
        )
        rejection_terms = (
            "does not support",
            "not supported",
            "unsupported",
            "unexpected keyword argument",
            "unknown parameter",
            "unrecognized parameter",
        )
        return (
            any(term in details for term in structured_terms)
            and any(term in details for term in rejection_terms)
        )

    @staticmethod
    def _structured_output_error(model: str) -> StructuredOutputCompatibilityError:
        return StructuredOutputCompatibilityError(
            f"The selected model or serving endpoint for '{model}' cannot provide the "
            "structured outputs Synthline requires. Choose a compatible model and endpoint "
            "with strict JSON Schema support, then try again."
        )

    @staticmethod
    def _empty_response_error(
        model: str,
        detail: str,
        *,
        structured: bool,
    ) -> Exception:
        if structured:
            return structured_output_response_error(model, StructuredOutputError(detail))
        return ProviderConfigurationError(
            f"The selected model or serving endpoint for '{model}' returned no usable "
            f"completion. {detail} Choose a compatible standard chat or instruct model, "
            "then try again."
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
        validate_model_compatibility(model, reasoning)
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

                if response_format:
                    kwargs["response_format"] = _response_format_for_model(
                        model_name,
                        response_format,
                    )
                if provider == "openrouter":
                    provider_options = kwargs.setdefault("extra_body", {}).setdefault(
                        "provider",
                        {},
                    )
                    provider_options["require_parameters"] = True

                response = await client.chat.completions.create(**kwargs)
                try:
                    message = response.choices[0].message
                except (AttributeError, IndexError, TypeError) as e:
                    raise self._empty_response_error(
                        model,
                        "The provider returned no completion choice.",
                        structured=response_format is not None,
                    ) from e
                completion_text = getattr(message, "content", None)
                if not isinstance(completion_text, str) or not completion_text.strip():
                    refusal = getattr(message, "refusal", None)
                    detail = f"The provider refused the request: {refusal}" if refusal else (
                        "The provider returned an empty completion."
                    )
                    raise self._empty_response_error(
                        model,
                        detail,
                        structured=response_format is not None,
                    )

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
                    if response_format and self._is_structured_output_compatibility_error(e):
                        self._logger.log_error(str(e), "llm", {"model": model})
                        raise self._structured_output_error(model) from e
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
                if response_format and self._is_structured_output_compatibility_error(e):
                    self._logger.log_error(str(e), "llm", {"model": model})
                    raise self._structured_output_error(model) from e
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
