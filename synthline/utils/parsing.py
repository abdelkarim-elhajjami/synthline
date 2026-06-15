"""
Parsing utilities for structured LLM outputs.
"""
import json
from typing import List

from synthline.errors import StructuredOutputError


def parse_completion(text: str, expected_count: int) -> List[str]:
    """Parse samples from LLM completion text.

    Args:
        text: LLM completion text (structured JSON).
        expected_count: Expected number of samples.

    Returns:
        List of sample texts.
    """
    requirement = "Synthline requires strict JSON Schema structured outputs."
    if not isinstance(text, str) or not text.strip():
        raise StructuredOutputError(f"The model returned no text. {requirement}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            f"The model returned invalid JSON instead of the required schema. {requirement}"
        ) from exc

    if not isinstance(data, dict) or set(data) != {"samples"}:
        raise StructuredOutputError(
            'Expected exactly one top-level "samples" field. '
            f"{requirement}"
        )

    samples = data["samples"]
    if not isinstance(samples, list):
        raise StructuredOutputError(
            'Expected "samples" to be an array of strings. '
            f"{requirement}"
        )

    if len(samples) != expected_count:
        raise StructuredOutputError(
            f"Expected exactly {expected_count} samples, received {len(samples)}. "
            f"{requirement}"
        )

    if any(not isinstance(item, str) or not item.strip() for item in samples):
        raise StructuredOutputError(
            'Expected every item in "samples" to be a non-empty string. '
            f"{requirement}"
        )

    return [item.strip() for item in samples]
