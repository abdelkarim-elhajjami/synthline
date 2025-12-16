import pytest
from unittest.mock import MagicMock, AsyncMock
from core.generator import Generator

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

@pytest.mark.asyncio
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
            "label": "User Story",
            "domain": "Healthcare",
            "language": "English",
            "stakeholder": "Patient",
            "specification_format": "User Story",
            "specification_level": "Detailed"
        }
        
        # Mock promptline to return one config
        config = features.copy()
        mock_promptline.get_atomic_configurations.return_value = [config]
        mock_promptline.build.return_value = "Generate detailed user stories for a healthcare system..."
        
        # Mock LLM to return valid JSON-like list of User Stories
        mock_response = '["As a patient, I want to view my medical records so that I can track my health history.", "As a patient, I want to book an appointment online to save time."]'
        mock_llm.get_batch_completions.return_value = [mock_response]
        
        # Act
        result = await generator.generate(features)
        samples = result["samples"]
        
        # Assert
        assert len(samples) == 2
        assert samples[0]["text"] == "As a patient, I want to view my medical records so that I can track my health history."
        assert samples[0]["label"] == "User Story"
        mock_llm.get_batch_completions.assert_called_once()

    asyncio.run(run())

@pytest.mark.asyncio
def test_generate_handles_llm_failure(generator, mock_llm, mock_logger, mock_promptline):
    """Test graceful handling of LLM errors."""
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
            "label": "Requirement", 
            "domain": "Fintech", 
            "language": "English", 
            "stakeholder": "Product Owner", 
            "specification_format": "Card", 
            "specification_level": "High"
        }
        mock_promptline.get_atomic_configurations.return_value = [full_features]
        mock_promptline.build.return_value = "Generate requirements..."
        
        # Mock LLM raising exception
        mock_llm.get_batch_completions.side_effect = Exception("API Rate Limit Exceeded")
        
        # Act
        result = await generator.generate(features)
        
        # Assert
        assert result["samples"] == []
        mock_logger.log_error.assert_called_once()
        assert "API Rate Limit Exceeded" in str(mock_logger.log_error.call_args)

    asyncio.run(run())

@pytest.mark.asyncio
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
            "label": "Use Case",
            "domain": "E-commerce",
            "language": "English",
            "stakeholder": "Shopper",
            "specification_format": "Use Case",
            "specification_level": "Detailed",
        }
        mock_promptline.get_atomic_configurations.return_value = [features.copy()]
        mock_promptline.build.return_value = "Generate use cases..."
        
        mock_response = '["Use Case: Checkout as Guest", "Use Case: Filter Products by Price"]'
        mock_llm.get_batch_completions.return_value = [mock_response]
        
        mock_callback = AsyncMock()
        
        # Act
        await generator.generate(features, progress_callback=mock_callback)
        
        # Assert
        # total_samples=2, samples_per_prompt=2 => 1 batch.
        # calls: start(0?) -> batch done -> finish(100). 
        # The generator implementation calls progress update per batch.
        assert mock_callback.call_count >= 1
        mock_callback.assert_called_with(100)

    asyncio.run(run())

@pytest.mark.asyncio
def test_generate_token_limit_handling(generator, mock_llm, mock_promptline, mock_logger):
    """Test handling of fewer samples received than requested (e.g. token limit cut off)."""
    import asyncio
    async def run():
        features = {
            "llm": "gpt-4",
            "temperature": 0.7,
            "top_p": 1.0,
            "total_samples": 2,
            "samples_per_prompt": 2,
            "label": "Requirement",
            "domain": "Cloud Infrastructure",
            "language": "English",
            "stakeholder": "DevOps Engineer",
            "specification_format": "Constrained NL",
            "specification_level": "Detailed"
        }
        mock_promptline.get_atomic_configurations.return_value = [features.copy()]
        mock_promptline.build.return_value = "Generate requirements..."
        
        # Return only 1 sample instead of 2 for the first call.
        # Then return the second sample in the subsequent call.
        mock_llm.get_batch_completions.side_effect = [
            ['["The system must auto-scale based on CPU usage."]'], # First call: returns 1 item (deficit)
            ['["The system must support multi-region deployment."]'] # Second call: returns 1 item (fill remainder)
        ]
        
        # Act
        result = await generator.generate(features)
        samples = result["samples"]
        
        # Assert
        # The generator should have made 2 calls and collected both samples.
        
        assert len(samples) == 2 
        assert samples[0]["text"] == "The system must auto-scale based on CPU usage."
        assert samples[1]["text"] == "The system must support multi-region deployment."
        
        mock_logger.log_error.assert_called()
        assert "Received fewer samples than requested" in str(mock_logger.log_error.call_args)
        
        # Verify multiple calls were made to fetch the remaining sample
        assert mock_llm.get_batch_completions.call_count == 2

    asyncio.run(run())
