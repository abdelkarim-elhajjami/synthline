import pytest
from unittest.mock import MagicMock, AsyncMock
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
        
        # Mock LLM to return valid JSON-like list of User Stories
        mock_response = '["As a patient, I want to view my medical records so that I can track my health history.", "As a patient, I want to book an appointment online to save time."]'
        mock_llm.get_batch_completions.return_value = [mock_response]
        
        # Act
        samples = await generator.generate(features)
        
        # Assert
        assert len(samples) == 2
        assert samples[0]["text"] == "As a patient, I want to view my medical records so that I can track my health history."
        assert samples[0]["config"]["classification_label"] == "User Story"
        mock_llm.get_batch_completions.assert_called_once()

    asyncio.run(run())

@pytest.mark.asyncio
def test_generate_handles_llm_failure(generator, mock_llm, mock_logger, mock_promptline):
    """Test fail-fast propagation for LLM errors."""
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
        
        with pytest.raises(FakeProviderError):
            await generator.generate(features)

        mock_logger.log_error.assert_called_once()
        assert "API request rejected" in str(mock_logger.log_error.call_args)

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
        
        mock_response = '["Use Case: Checkout as Guest", "Use Case: Filter Products by Price"]'
        mock_llm.get_batch_completions.return_value = [mock_response]
        
        mock_callback = AsyncMock()
        
        # Act
        await generator.generate(features, progress_callback=mock_callback)
        
        # Assert
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
        
        # Return only 1 sample instead of 2 for the first call.
        mock_llm.get_batch_completions.side_effect = [
            ['["The system must auto-scale based on CPU usage."]'],
            ['["The system must support multi-region deployment."]']
        ]
        
        # Act
        samples = await generator.generate(features)
        
        # Assert
        assert len(samples) == 2 
        assert samples[0]["text"] == "The system must auto-scale based on CPU usage."
        assert samples[1]["text"] == "The system must support multi-region deployment."
        
        mock_logger.log_error.assert_called()
        assert "Received fewer samples than requested" in str(mock_logger.log_error.call_args)
        
        # Verify multiple calls were made to fetch the remaining sample
        assert mock_llm.get_batch_completions.call_count == 2

    asyncio.run(run())
