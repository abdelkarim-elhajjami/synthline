"""
Parsing utilities for structured LLM outputs.
"""
import json
from typing import List

from synthline.types import ParseResult


def parse_completion(text: str, expected_count: int) -> List[str]:
    """Parse samples from LLM completion text.

    Args:
        text: LLM completion text (structured JSON).
        expected_count: Expected number of samples.

    Returns:
        List of sample texts.
    """
    return parse_completion_with_meta(text, expected_count).samples


def parse_completion_with_meta(text: str, expected_count: int) -> ParseResult:
    """Parse completion into a :class:`ParseResult`.

    ``method`` is one of:
    - ``"json"``      – valid JSON parsed successfully
    - ``"plaintext"`` – JSON parsing failed; raw text returned as-is
    """
    stripped = text.strip()

    # Try structured output: {"samples": ["...", "..."]}
    samples = _try_unwrap_structured(stripped)
    if samples:
        return ParseResult(samples=samples, degraded=False, method="json")

    # Fallback: raw text as single sample.
    # Degraded if we expected structured output (multiple samples).
    return ParseResult(samples=[stripped], degraded=expected_count > 1, method="plaintext")


def _try_unwrap_structured(text: str) -> List[str]:
    """Unwrap structured output: {"samples": ["...", ...]} → list of strings."""
    if not text.startswith('{'):
        return []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Prefer the "samples" key (matches samples_schema()), then fall back to any key
            candidates = [data["samples"]] if "samples" in data else data.values()
            for value in candidates:
                if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
                    return [item.strip() for item in value if item.strip()]
    except Exception:
        pass  # Malformed JSON — fall through to plaintext path
    return []
