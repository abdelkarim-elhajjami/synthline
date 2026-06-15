from synthline.client import Synthline
from synthline.errors import (
    AlignmentVerificationError,
    ProviderConfigurationError,
    StructuredOutputCompatibilityError,
    StructuredOutputError,
    StructuredOutputResponseError,
    SynthlineError,
)
from synthline.types import (
    Dataset,
    ProgressCallback,
    PromptEntry,
    PromptSet,
    PromptUpdateCallback,
    PromptUpdateEvent,
    VerificationCallback,
    VerificationEvent,
)

__version__ = "0.3.0"
__all__ = [
    "Synthline",
    "SynthlineError",
    "AlignmentVerificationError",
    "ProviderConfigurationError",
    "StructuredOutputError",
    "StructuredOutputCompatibilityError",
    "StructuredOutputResponseError",
    "Dataset",
    "ProgressCallback",
    "PromptEntry",
    "PromptSet",
    "PromptUpdateCallback",
    "PromptUpdateEvent",
    "VerificationCallback",
    "VerificationEvent",
    "__version__",
]
