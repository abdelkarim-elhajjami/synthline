"""Public Synthline exception types."""


class SynthlineError(Exception):
    """Base class for actionable errors safe to show to SDK users."""


class StructuredOutputError(SynthlineError):
    """Raised when a response does not satisfy Synthline's required schema."""


class StructuredOutputCompatibilityError(StructuredOutputError):
    """Raised when a model or serving endpoint rejects structured outputs."""


class StructuredOutputResponseError(StructuredOutputError):
    """Raised when a completion violates Synthline's required output schema."""


class ProviderConfigurationError(SynthlineError):
    """Raised when a selected LLM provider is not configured correctly."""


class AlignmentVerificationError(SynthlineError):
    """Raised when alignment verification cannot score generated samples."""


def structured_output_response_error(
    model: str,
    cause: StructuredOutputError,
) -> StructuredOutputResponseError:
    """Build an actionable error for a response that violated the required schema."""
    return StructuredOutputResponseError(
        f"The selected model or serving endpoint for '{model}' did not return the "
        "structured output Synthline requires. The response may have been truncated, "
        "refused, or malformed. Check the model response and endpoint compatibility, "
        "then try again. "
        f"Details: {cause}"
    )
