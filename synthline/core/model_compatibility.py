"""Model compatibility rules shared by the SDK and Web API."""

from typing import Any, Dict, Optional

from synthline.errors import ProviderConfigurationError


def is_reasoning_model(model: str) -> bool:
    """Return whether *model* belongs to a known reasoning-style family."""
    model_name = str(model).strip().lower().replace("_", "-").split("/")[-1]
    return (
        model_name.startswith(("deepseek-r1", "magistral", "qwq"))
        or model_name.startswith(tuple(f"o{i}" for i in range(1, 10)))
        or model_name.startswith("gpt-5")
        or "thinking" in model_name
        or "reasoning" in model_name
        or "reasoner" in model_name
    )


def validate_model_compatibility(
    model: str,
    reasoning: Optional[Dict[str, Any]] = None,
) -> None:
    """Reject models and options outside Synthline's generation contract."""
    if reasoning:
        raise ProviderConfigurationError(
            "Synthline does not support reasoning options. Synthetic data generation "
            "requires predictable, high-throughput sampling with temperature and top_p. "
            "Remove the reasoning configuration and choose a standard chat or instruct model "
            "with strict JSON Schema structured-output support."
        )
    if is_reasoning_model(model):
        raise ProviderConfigurationError(
            f"Synthline does not support the reasoning-style model '{model}'. Reasoning models "
            "are slower, costlier, and often reject the sampling controls Synthline uses for "
            "synthetic data generation. Choose a standard chat or instruct model with strict "
            "JSON Schema structured-output support."
        )
