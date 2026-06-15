from synthline.errors import (
    AlignmentVerificationError,
    ProviderConfigurationError,
    StructuredOutputCompatibilityError,
    StructuredOutputError,
)
from utils.errors import operation_error_message


def test_structured_output_errors_are_returned_without_technical_prefix():
    compatibility = StructuredOutputCompatibilityError("Choose a compatible model.")
    validation = StructuredOutputError("The model returned the wrong schema.")

    assert operation_error_message("generation", compatibility) == str(compatibility)
    assert operation_error_message("optimization", validation) == str(validation)


def test_provider_and_verification_errors_are_returned_without_technical_prefix():
    provider = ProviderConfigurationError("Set OLLAMA_BASE_URL.")
    verification = AlignmentVerificationError("NLI model failed.")

    assert operation_error_message("generation", provider) == str(provider)
    assert operation_error_message("generation", verification) == str(verification)


def test_unexpected_errors_keep_operation_context():
    assert operation_error_message("generation", RuntimeError("Connection lost")) == (
        "Generation error: Connection lost"
    )
