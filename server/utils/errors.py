"""User-facing error formatting for WebSocket operations."""

from synthline.errors import SynthlineError


def operation_error_message(operation: str, error: Exception) -> str:
    """Keep known actionable errors clean while labeling unexpected failures."""
    if isinstance(error, SynthlineError):
        return str(error)
    return f"{operation.capitalize()} error: {error}"
