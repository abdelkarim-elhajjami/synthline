import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from synthline.core.generator import Generator

@pytest.fixture
def mock_llm():
    return AsyncMock()

@pytest.fixture
def mock_promptline():
    return MagicMock()

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def generator(mock_llm, mock_promptline, mock_logger):
    return Generator(mock_llm, mock_promptline, mock_logger)

def test_generate_success(generator, mock_llm, mock_promptline):
    """Test standard generation flow."""
    import asyncio
    async def run():
        features = {
            "llm": "gpt-4",
            "temperature": 0.7,
            "top_p": 1.0,
            "total_samples": 2,
            "samples_per_prompt": 2,
            "classification_label": "User Story",
            "classification_label_def": "A user story requirement",
            "requirement_type": "Functional",
            "requirement_subtype": "FunctionalPerspective",
            "domain": "Healthcare",
            "language": "English",
            "stakeholder_viewpoint": "Patient",
            "context": "Usage",
            "abstraction_level": "DetailedLevel"
        }

        # Mock promptline to return one config
        config = features.copy()
        mock_promptline.get_atomic_configurations.return_value = [config]
        mock_promptline.build.return_value = "Generate detailed user stories for a healthcare system..."

        mock_response = '{"samples": ["As a patient, I want to view my medical records so that I can track my health history.", "As a patient, I want to book an appointment online to save time."]}'
        mock_llm.get_batch_completions.return_value = [mock_response]

        # Act
        result = await generator.generate(features)

        # Assert
        assert len(result.samples) == 2
        assert result.samples[0]["text"] == "As a patient, I want to view my medical records so that I can track my health history."
        assert result.samples[0]["config"]["classification_label"] == "User Story"
        mock_llm.get_batch_completions.assert_called_once()

    asyncio.run(run())

def test_generate_handles_llm_failure(generator, mock_llm, mock_logger, mock_promptline):
    """Test graceful degradation on LLM errors (returns empty, no crash)."""
    import asyncio
    async def run():
        features = {
            "llm": "gpt-4",
            "temperature": 0.7,
            "top_p": 1.0,
            "total_samples": 1,
            "samples_per_prompt": 1
        }
        full_features = {
            **features,
            "classification_label": "Requirement",
            "classification_label_def": "A system requirement",
            "requirement_type": "Functional",
            "requirement_subtype": "DataPerspective",
            "domain": "Fintech",
            "language": "English",
            "stakeholder_viewpoint": "Product Owner",
            "context": "Business",
            "abstraction_level": "HighLevel"
        }
        mock_promptline.get_atomic_configurations.return_value = [full_features]
        mock_promptline.build.return_value = "Generate requirements..."

        class FakeProviderError(Exception):
            def __init__(self):
                super().__init__("API request rejected")
                self.status_code = 429
                self.body = {"error": {"type": "rate_limit"}}

        mock_llm.get_batch_completions.side_effect = FakeProviderError()

        result = await generator.generate(features)
        assert result.samples == []
        assert result.fewer_samples_received is True

    asyncio.run(run())

def test_progress_reporting(generator, mock_llm, mock_promptline):
    """Test progress callback is invoked with realistic data."""
    import asyncio
    async def run():
        features = {
            "llm": "gpt-4",
            "temperature": 0.7,
            "top_p": 1.0,
            "total_samples": 2,
            "samples_per_prompt": 2,
            "classification_label": "Use Case",
            "classification_label_def": "A use case description",
            "requirement_type": "Functional",
            "requirement_subtype": "BehavioralPerspective",
            "domain": "E-commerce",
            "language": "English",
            "stakeholder_viewpoint": "Shopper",
            "context": "Usage",
            "abstraction_level": "DetailedLevel",
        }
        mock_promptline.get_atomic_configurations.return_value = [features.copy()]
        mock_promptline.build.return_value = "Generate use cases..."

        mock_response = '{"samples": ["Use Case: Checkout as Guest", "Use Case: Filter Products by Price"]}'
        mock_llm.get_batch_completions.return_value = [mock_response]

        mock_callback = AsyncMock()

        # Act
        await generator.generate(features, progress_callback=mock_callback)

        # Assert
        assert mock_callback.call_count >= 1
        mock_callback.assert_called_with(100, "Generation complete")

    asyncio.run(run())

def test_generate_token_limit_handling(generator, mock_llm, mock_promptline, mock_logger):
    """Test that valid JSON with short count is accepted (not retried),
    and the loop fills the remaining samples on the next call."""
    import asyncio
    async def run():
        features = {
            "llm": "gpt-4",
            "temperature": 0.7,
            "top_p": 1.0,
            "total_samples": 2,
            "samples_per_prompt": 2,
            "classification_label": "Requirement",
            "classification_label_def": "A cloud requirement",
            "requirement_type": "Quality",
            "requirement_subtype": "PerformanceEfficiency",
            "domain": "Cloud Infrastructure",
            "language": "English",
            "stakeholder_viewpoint": "DevOps Engineer",
            "context": "Technical",
            "abstraction_level": "DetailedLevel"
        }
        mock_promptline.get_atomic_configurations.return_value = [features.copy()]
        mock_promptline.build.return_value = "Generate requirements..."

        # First call returns 1/2 as valid JSON → accepted (not retried).
        # Second call fills the remaining 1.
        mock_llm.get_batch_completions.side_effect = [
            ['{"samples": ["The system must auto-scale based on CPU usage."]}'],
            ['{"samples": ["The system must support multi-region deployment."]}'],
        ]

        # Act
        result = await generator.generate(features)

        # Assert — both samples collected across two calls
        assert len(result.samples) == 2
        assert result.samples[0]["text"] == "The system must auto-scale based on CPU usage."
        assert result.samples[1]["text"] == "The system must support multi-region deployment."

        # No warning — valid JSON is accepted silently
        mock_logger.log_warning.assert_not_called()

        # Two calls: first returned 1, second filled the gap
        assert mock_llm.get_batch_completions.call_count == 2

    asyncio.run(run())


# ======================================================================
# generate_for_configs tests
# ======================================================================

def _make_config(config_id: str, *, optimized: bool = False) -> dict:
    """Helper to build a distinct config for testing."""
    config = {
        "llm": "test-model",
        "temperature": 0.7,
        "top_p": 1.0,
        "__fm_constraints__": [{"label": f"Feature_{config_id}", "value": config_id}],
        "classification_label": "Test",
        "classification_label_def": "A test label",
        "prompt": f"Generate test data for config {config_id}",
    }
    if optimized:
        config["optimized_prompt"] = f"Optimized prompt for config {config_id}"
    return config


def test_generate_for_configs_single_config(generator, mock_llm):
    """Single config, 2 samples needed."""
    import asyncio

    async def run():
        config = _make_config("A")
        mock_llm.get_batch_completions.return_value = [
            '{"samples": ["Sample 1", "Sample 2"]}'
        ]

        result = await generator.generate_for_configs(
            config_requests=[(config, 2)],
            samples_per_prompt=2,
        )

        assert len(result.samples) == 2
        assert result.samples[0]["text"] == "Sample 1"
        assert result.fewer_samples_received is False
        mock_llm.get_batch_completions.assert_called_once()

    asyncio.run(run())


def test_generate_for_configs_multi_config(generator, mock_llm):
    """Two configs with different counts, each gets its own LLM call."""
    import asyncio

    async def run():
        config_a = _make_config("A")
        config_b = _make_config("B")

        mock_llm.get_batch_completions.side_effect = [
            ['{"samples": ["A1"]}'],
            ['{"samples": ["B1", "B2"]}'],
        ]

        result = await generator.generate_for_configs(
            config_requests=[(config_a, 1), (config_b, 2)],
            samples_per_prompt=2,
        )

        assert len(result.samples) == 3
        assert result.samples[0]["text"] == "A1"
        assert result.samples[1]["text"] == "B1"
        assert mock_llm.get_batch_completions.call_count == 2

    asyncio.run(run())


def test_generate_for_configs_uses_optimized_prompt(generator, mock_llm):
    """Config with optimized_prompt should use it instead of promptline.build."""
    import asyncio

    async def run():
        config = _make_config("A", optimized=True)

        mock_llm.get_batch_completions.return_value = [
            '{"samples": ["Sample 1"]}'
        ]

        result = await generator.generate_for_configs(
            config_requests=[(config, 1)],
            samples_per_prompt=1,
        )

        assert len(result.samples) == 1
        # The prompt in the sample config should be the optimized one
        sample_config = result.samples[0]["config"]
        assert sample_config["prompt"] == "Optimized prompt for config A"

    asyncio.run(run())


def test_generate_for_configs_sets_optimized_prompt_from_stored(
    generator, mock_llm, mock_promptline
):
    """Non-PACE config: optimized_prompt is set from stored prompt to bypass
    promptline.build and reuse the exact same prompt."""
    import asyncio

    async def run():
        config = _make_config("A")  # No optimized_prompt
        assert "optimized_prompt" not in config

        mock_llm.get_batch_completions.return_value = [
            '{"samples": ["Sample 1"]}'
        ]

        result = await generator.generate_for_configs(
            config_requests=[(config, 1)],
            samples_per_prompt=1,
        )

        assert len(result.samples) == 1
        # promptline.build should NOT have been called
        mock_promptline.build.assert_not_called()

    asyncio.run(run())


def test_generate_for_configs_handles_llm_failure(generator, mock_llm, mock_logger):
    """One config fails, other succeeds — partial results returned."""
    import asyncio

    async def run():
        config_a = _make_config("A")
        config_b = _make_config("B")

        mock_llm.get_batch_completions.side_effect = [
            RuntimeError("Provider failed"),
            ['{"samples": ["B1"]}'],
        ]

        result = await generator.generate_for_configs(
            config_requests=[(config_a, 1), (config_b, 1)],
            samples_per_prompt=1,
        )

        assert len(result.samples) == 1
        assert result.samples[0]["text"] == "B1"
        assert result.fewer_samples_received is True

    asyncio.run(run())


def test_generate_for_configs_multi_call_per_config(generator, mock_llm):
    """Config needs 4 samples with spp=2 → 2 LLM calls."""
    import asyncio

    async def run():
        config = _make_config("A")

        mock_llm.get_batch_completions.side_effect = [
            ['{"samples": ["S1", "S2"]}'],
            ['{"samples": ["S3", "S4"]}'],
        ]

        result = await generator.generate_for_configs(
            config_requests=[(config, 4)],
            samples_per_prompt=2,
        )

        assert len(result.samples) == 4
        assert mock_llm.get_batch_completions.call_count == 2

    asyncio.run(run())


def test_generate_for_configs_empty_request(generator):
    """Empty config_requests → empty result."""
    import asyncio

    async def run():
        result = await generator.generate_for_configs(
            config_requests=[],
            samples_per_prompt=2,
        )

        assert result.samples == []
        assert result.fewer_samples_received is False

    asyncio.run(run())
