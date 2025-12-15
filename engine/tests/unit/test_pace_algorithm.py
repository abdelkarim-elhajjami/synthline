import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from core.pace import PACE

@pytest.fixture
def mock_llm():
    return AsyncMock()

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    with patch("core.pace.SentenceTransformer") as mock_st_cls:
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float) 
        yield mock_model

@pytest.fixture
def pace_instance(mock_llm, mock_logger):
    return PACE(mock_llm, mock_logger)

@pytest.mark.asyncio
def test_optimize_batch_flow(pace_instance, mock_llm, mock_sentence_transformer):
    """Test the high-level batch optimization flow."""
    import asyncio
    async def run():
        # Arrange
        atomic_configs = [{"prompt": "Write a user story for login", "label": "Security"}]
        features = {
            "domain": "Cybersecurity",
            "label": "Security",
            "label_definition": "A security constraint",
            "language": "English",
            "stakeholder": "Security Engineer",
            "specification_format": "User Story",
            "specification_level": "Detailed",
            "samples_per_prompt": 2,
            "llm": "gpt-4"
        }
        
        actor_response = '["As a user, I want 2FA enabled to secure my account.", "As an admin, I want to enforce password complexity."]'
        critic_response = "Critique: The stories lack acceptance criteria."
        update_response = "Write a user story for login including acceptance criteria"
        
        # Setup LLM side effects for sequence of calls
        # Call 1 (Actor): Initial Prompt -> Samples
        # Call 2 (Critic): Samples -> Critique
        # Call 3 (Update): Critique -> New Prompt
        # Call 4 (Actor - Candidate Evaluation): New Prompt -> Samples
        mock_llm.get_batch_completions.side_effect = [
            [actor_response],
            [critic_response],
            [update_response],
            [actor_response]
        ]
        
        results = await pace_instance.optimize_batch(
            atomic_configs, 
            features, 
            n_iterations=1, 
            n_actors=1, 
            n_candidates=1
        )
        
        # Assert
        assert len(results) == 1
        best_prompt, best_score, _ = results[0]
        # Assert best prompt
        assert best_prompt == "Write a user story for login including acceptance criteria"
        # Assert best score
        assert isinstance(best_score, float)
        # Assert LLM call count
        assert mock_llm.get_batch_completions.call_count >= 4
        
        # Verify that SentenceTransformer was indeed used
        mock_sentence_transformer.encode.assert_called()

    asyncio.run(run())

def test_evaluate_prompt_logic(pace_instance):
    """Test the scoring logic separately."""
    # Arrange
    # 2 samples, orthogonal => distance should be > 0.
    # Raw completion containing typical JSON
    raw_completion = '["Requirement 1", "Requirement 2"]'
    
    # Act
    score = pace_instance._evaluate_prompt([raw_completion], samples_per_prompt=2)
    
    # Assert
    assert score > 0.0
    
    # Test failure case: Invalid JSON
    score_invalid = pace_instance._evaluate_prompt(["Invalid JSON"], samples_per_prompt=2)
    assert score_invalid == 0.0

@pytest.mark.asyncio
def test_update_prompt_handles_failure(pace_instance, mock_llm, mock_logger):
    """Test that if update prompt fails, we keep the current prompt."""
    import asyncio
    async def run():
        mock_llm.get_batch_completions.side_effect = Exception("Update failed")
        
        updated_prompt = await pace_instance._update_prompt("Initial Prompt", ["Feedback"], {"llm": "gpt-4"})
        
        assert updated_prompt == "Initial Prompt"
        mock_logger.log_error.assert_called()

    asyncio.run(run())
