"""Shared structured output schemas for LLM calls."""
from typing import Any, Dict

def samples_schema(count: int) -> Dict[str, Any]:
    """Build a structured output schema that enforces exactly *count* items."""
    if count < 1:
        raise ValueError("Structured sample count must be at least 1.")

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "samples",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "samples": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": count,
                        "maxItems": count,
                    }
                },
                "required": ["samples"],
                "additionalProperties": False,
            },
        },
    }
