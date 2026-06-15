import pytest

from synthline.core.model_compatibility import (
    is_reasoning_model,
    validate_model_compatibility,
)
from synthline.errors import ProviderConfigurationError


@pytest.mark.parametrize(
    "model",
    [
        "openrouter/mistralai/ministral-8b",
        "openrouter/anthropic/claude-sonnet",
        "openai/gpt-4.1-mini",
        "ollama/llama3.2",
        "huggingface/meta-llama/Llama-3.3-70B-Instruct",
    ],
)
def test_standard_chat_models_are_allowed(model):
    assert is_reasoning_model(model) is False
    validate_model_compatibility(model)


def test_explicit_reasoning_configuration_is_rejected():
    with pytest.raises(ProviderConfigurationError, match="does not support reasoning options"):
        validate_model_compatibility(
            "openrouter/mistralai/ministral-8b",
            {"effort": "high"},
        )
