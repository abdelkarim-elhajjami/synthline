import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from huggingface_hub import AsyncInferenceClient
from openai.resources.chat.completions import AsyncCompletions

from synthline.core.llm import LLMClient
from synthline.core.schemas import samples_schema
from synthline.errors import ProviderConfigurationError, StructuredOutputCompatibilityError
from synthline.errors import StructuredOutputResponseError


def _completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=None))]
    )


@pytest.mark.parametrize(
    "model",
    [
        "ollama/test-model",
        "huggingface/org/test-model",
        "openai/test-model",
        "openrouter/google/test-model",
        "ilaas/test-model",
    ],
)
def test_providers_use_shared_openai_compatible_response_format(model):
    async def run():
        logger = MagicMock()
        llm = LLMClient(logger=logger)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_completion('{"samples": ["A"]}')
        )
        llm._get_client = MagicMock(return_value=client)
        response_format = samples_schema(1)

        result = await llm.get_completion(
            prompt="Generate one sample",
            model=model,
            temperature=0.0,
            top_p=1.0,
            response_format=response_format,
        )

        assert result == '{"samples": ["A"]}'
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == response_format
        assert "format" not in kwargs

    asyncio.run(run())


def test_namespaced_ollama_model_name_is_preserved():
    assert LLMClient._model_name_for_request("ollama/acme/custom-model:latest") == (
        "acme/custom-model:latest"
    )


def test_ollama_requires_explicit_configuration():
    llm = LLMClient(logger=MagicMock())

    with pytest.raises(ProviderConfigurationError, match="OLLAMA_BASE_URL"):
        llm._get_client("ollama/test-model")


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("http://localhost:11434", "http://localhost:11434/v1"),
        ("http://localhost:11434/", "http://localhost:11434/v1"),
        ("http://localhost:11434/v1", "http://localhost:11434/v1"),
    ],
)
def test_ollama_base_url_is_normalized(configured, expected):
    llm = LLMClient(logger=MagicMock(), ollama_base_url=configured)
    llm._create_async_client = MagicMock(return_value=MagicMock())

    llm._get_client("ollama/test-model")

    llm._create_async_client.assert_called_once_with(
        base_url=expected,
        api_key="dummy",
    )


@pytest.mark.parametrize(
    "model",
    [
        "openai/o3-mini",
        "openrouter/openai/gpt-5",
        "openrouter/deepseek/deepseek-r1",
        "huggingface/Qwen/Qwen3-Thinking",
        "ollama/magistral",
    ],
)
def test_reasoning_models_are_rejected_before_provider_call(model):
    async def run():
        llm = LLMClient(logger=MagicMock())
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        llm._get_client = MagicMock(return_value=client)

        with pytest.raises(ProviderConfigurationError, match="does not support the reasoning-style model"):
            await llm.get_completion(
                prompt="Generate one sample",
                model=model,
                temperature=1.0,
                top_p=1.0,
                response_format=samples_schema(1),
            )

        llm._get_client.assert_not_called()
        client.chat.completions.create.assert_not_awaited()
    asyncio.run(run())


def test_reasoning_options_are_rejected_before_provider_call():
    async def run():
        llm = LLMClient(logger=MagicMock())
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        llm._get_client = MagicMock(return_value=client)

        with pytest.raises(ProviderConfigurationError, match="does not support reasoning options"):
            await llm.get_completion(
                prompt="Generate one sample",
                model="openrouter/google/test-model",
                temperature=0.5,
                top_p=0.9,
                reasoning={"effort": "high"},
            )

        llm._get_client.assert_not_called()
        client.chat.completions.create.assert_not_awaited()

    asyncio.run(run())


def test_installed_dependency_apis_support_required_request_contract():
    openai_parameters = inspect.signature(AsyncCompletions.create).parameters
    hf_parameters = inspect.signature(
        AsyncInferenceClient().chat.completions.create
    ).parameters

    assert {
        "response_format",
        "max_tokens",
    } <= set(
        openai_parameters
    )
    assert {"response_format", "max_tokens"} <= set(hf_parameters)


def test_huggingface_client_builds_enforced_structured_output_request():
    async def run():
        hf_client = AsyncInferenceClient(
            base_url="https://example.test/v1",
            api_key="test",
        )
        hf_client._inner_post = AsyncMock(
            return_value={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "org/test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"samples":["A"]}',
                        },
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )
        llm = LLMClient(logger=MagicMock())
        llm._hf_client = hf_client
        response_format = samples_schema(1)

        result = await llm.get_completion(
            prompt="Generate one sample",
            model="huggingface/org/test-model",
            temperature=0.0,
            top_p=1.0,
            response_format=response_format,
        )

        request = hf_client._inner_post.await_args.args[0]
        sent_format = request.json["response_format"]
        assert result == '{"samples":["A"]}'
        if sent_format["type"] == "json_schema":
            assert sent_format == response_format
        else:
            # huggingface_hub 0.32.6 translates the OpenAI-compatible schema
            # into its equivalent enforced JSON-object grammar.
            assert sent_format == {
                "type": "json_object",
                "value": response_format["json_schema"]["schema"],
            }

    asyncio.run(run())


def test_openrouter_requires_a_provider_that_supports_every_sent_parameter():
    async def run():
        logger = MagicMock()
        llm = LLMClient(logger=logger)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_completion('{"samples": ["A"]}')
        )
        llm._get_client = MagicMock(return_value=client)

        await llm.get_completion(
            prompt="Generate one sample",
            model="openrouter/google/test-model",
            temperature=0.0,
            top_p=1.0,
            response_format=samples_schema(1),
        )

        kwargs = client.chat.completions.create.call_args.kwargs
        extra_body = kwargs["extra_body"]
        assert extra_body["provider"] == {"require_parameters": True}
        assert kwargs["temperature"] == 0.0
        assert kwargs["top_p"] == 1.0
        assert kwargs["max_tokens"] == LLMClient.DEFAULT_MAX_TOKENS

    asyncio.run(run())


def test_openrouter_unstructured_calls_also_require_supported_parameters():
    async def run():
        llm = LLMClient(logger=MagicMock())
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_completion("Useful critique"))
        llm._get_client = MagicMock(return_value=client)

        await llm.get_completion(
            prompt="Critique this prompt",
            model="openrouter/google/test-model",
            temperature=0.3,
            top_p=0.9,
        )

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"provider": {"require_parameters": True}}
        assert "response_format" not in kwargs

    asyncio.run(run())


def test_anthropic_schema_normalization_preserves_shared_contract():
    async def run():
        logger = MagicMock()
        llm = LLMClient(logger=logger)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_completion('{"samples": ["A"]}')
        )
        llm._get_client = MagicMock(return_value=client)
        response_format = samples_schema(1)

        await llm.get_completion(
            prompt="Generate one sample",
            model="openrouter/anthropic/claude-sonnet",
            temperature=0.0,
            top_p=1.0,
            response_format=response_format,
        )

        sent = client.chat.completions.create.call_args.kwargs["response_format"]
        sent_samples = sent["json_schema"]["schema"]["properties"]["samples"]
        original_samples = response_format["json_schema"]["schema"]["properties"]["samples"]
        assert "minItems" not in sent_samples
        assert "maxItems" not in sent_samples
        assert original_samples["minItems"] == 1
        assert original_samples["maxItems"] == 1

    asyncio.run(run())


def test_internal_client_argument_error_is_not_blame_shifted_to_model():
    async def run():
        logger = MagicMock()
        llm = LLMClient(logger=logger)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=TypeError(
                "AsyncCompletions.create() got an unexpected keyword argument 'format'"
            )
        )
        llm._get_client = MagicMock(return_value=client)

        with pytest.raises(TypeError, match="unexpected keyword argument 'format'"):
            await llm.get_completion(
                prompt="Generate one sample",
                model="ollama/test-model",
                temperature=0.0,
                top_p=1.0,
                response_format=samples_schema(1),
            )

    asyncio.run(run())


def test_provider_structured_output_rejection_is_user_facing():
    async def run():
        logger = MagicMock()
        llm = LLMClient(logger=logger)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("response_format json_schema is not supported by this model")
        )
        llm._get_client = MagicMock(return_value=client)

        with pytest.raises(StructuredOutputCompatibilityError) as exc_info:
            await llm.get_completion(
                prompt="Generate one sample",
                model="openai/unsupported-model",
                temperature=0.0,
                top_p=1.0,
                response_format=samples_schema(1),
            )

        message = str(exc_info.value)
        assert "The selected model or serving endpoint for 'openai/unsupported-model'" in message
        assert "response_format" not in message

    asyncio.run(run())


def test_invalid_schema_error_is_not_misreported_as_model_incompatibility():
    error = RuntimeError("Invalid json_schema: required field is missing")

    assert LLMClient._is_structured_output_compatibility_error(error) is False


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=None),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace())]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", refusal=None))]),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None, refusal="Safety refusal")
                )
            ]
        ),
    ],
)
def test_empty_or_refused_structured_response_is_user_facing(response):
    async def run():
        llm = LLMClient(logger=MagicMock())
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        llm._get_client = MagicMock(return_value=client)

        with pytest.raises(
            StructuredOutputResponseError,
            match="did not return the structured output Synthline requires",
        ):
            await llm.get_completion(
                prompt="Generate one sample",
                model="openai/test-model",
                temperature=0.0,
                top_p=1.0,
                response_format=samples_schema(1),
            )

    asyncio.run(run())


def test_empty_unstructured_response_is_user_facing():
    async def run():
        llm = LLMClient(logger=MagicMock())
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None, refusal="Request refused")
                    )
                ]
            )
        )
        llm._get_client = MagicMock(return_value=client)

        with pytest.raises(ProviderConfigurationError, match="returned no usable completion"):
            await llm.get_completion(
                prompt="Critique this prompt",
                model="openai/test-model",
                temperature=0.0,
                top_p=1.0,
            )

    asyncio.run(run())
