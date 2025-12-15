import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from core.promptline import Promptline

@pytest.fixture
def mock_llm():
    return MagicMock()

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def promptline(mock_llm, mock_logger):
    return Promptline(mock_llm, mock_logger)

def test_get_atomic_configurations_basic(promptline):
    """Test standard feature combination logic."""
    features = {
        "generator": {"llm": "gpt-4"},
        "specification_format": ["User Story", "Use Case"],
        "specification_level": ["High"],
        "stakeholder": "User",
        "domain": "Retail, Finance",
        "language": "English",
        "samples_per_prompt": 1
    }
    
    configs = promptline.get_atomic_configurations(features)
    
    # Combinations: 2 formats * 1 level * 1 stakeholder * 2 domains * 1 language = 4 configs
    assert len(configs) == 4
    
    # Verify content of one config
    domains = [c['domain'] for c in configs]
    assert "Retail" in domains
    assert "Finance" in domains
    
    formats = [c['specification_format'] for c in configs]
    assert "User Story" in formats
    assert "Use Case" in formats

def test_get_atomic_configurations_single_value(promptline):
    """Test when no multi-select features are present."""
    features = {
        "specification_format": "User Story",
        "domain": "Retail",
        "samples_per_prompt": 1
    }
    
    configs = promptline.get_atomic_configurations(features)
    assert len(configs) == 1
    assert configs[0]['specification_format'] == "User Story"

def test_build_prompt_single_sample(promptline):
    """Test prompt generation for single sample."""
    features = {
        "label": "Bug",
        "label_definition": "An error",
        "language": "English",
        "domain": "Web",
        "stakeholder": "Tester",
        "specification_format": "Bug Report",
        "specification_level": "Detailed",
        "samples_per_prompt": 1
    }
    
    prompt = promptline.build(features)
    assert "Generate a requirement that" in prompt
    assert "An error" in prompt
    assert "Bug Report" in prompt

def test_build_prompt_multi_sample(promptline):
    """Test prompt generation for multiple samples."""
    features = {
        "label": "Bug",
        "label_definition": "An error",
        "language": "English",
        "domain": "Web",
        "stakeholder": "Tester",
        "specification_format": "Bug Report",
        "specification_level": "Detailed",
        "samples_per_prompt": 5
    }
    
    prompt = promptline.build(features)
    assert "Generate 5 diverse requirements" in prompt
    assert "JSON array" in prompt

def test_optimize_batch_calls_pace(promptline):
    """Test that optimize_batch correctly delegates to PACE."""
    import asyncio
    
    features = {
        "pace_iterations": 2,
        "pace_actors": 2,
        "pace_candidates": 2
    }
    atomic_configs = [{"prompt": "Generate a user story for login", "label": "User Story"}]
    
    async def run_test():
        # Mock PACE class used inside promptline
        with patch("core.promptline.PACE") as MockPACE:
            mock_pace_instance = MockPACE.return_value
            mock_pace_instance.optimize_batch = AsyncMock(return_value=["result"])
            
            await promptline.optimize_batch(atomic_configs, features)
            
            # Verify PACE was initialized
            MockPACE.assert_called_once()
            
            # Verify optimize_batch was called with correct args
            mock_pace_instance.optimize_batch.assert_called_once()
            call_kwargs = mock_pace_instance.optimize_batch.call_args.kwargs
            assert call_kwargs['n_iterations'] == 2
            assert call_kwargs['atomic_configs'] == atomic_configs

    asyncio.run(run_test())

def test_get_atomic_prompts(promptline):
    """Test generating full atomic prompt objects."""
    features = {
        "specification_format": ["User Story", "Gherkin"],
        "domain": "E-commerce",
        "samples_per_prompt": 1,
        "label": "Feature",
        "label_definition": "A functional requirement",
        "language": "English",
        "stakeholder": "Product Owner",
        "specification_level": "High-level"
    }
    
    atomic_prompts = promptline.get_atomic_prompts(features)
    
    assert len(atomic_prompts) == 2
    
    # Verify structure
    assert "config" in atomic_prompts[0]
    assert "prompt" in atomic_prompts[0]
    
    # Verify content varies
    formats = [p['config']['specification_format'] for p in atomic_prompts]
    assert "User Story" in formats
    assert "Gherkin" in formats
    
    # Verify prompts are generated
    assert all(isinstance(p['prompt'], str) for p in atomic_prompts)
    assert all(len(p['prompt']) > 0 for p in atomic_prompts)

def test_build_prompt_error_handling(promptline, mock_logger):
    """Test that build raises exception and logs error on failure."""
    # Missing required keys for the template
    features = {
        "samples_per_prompt": 1
    }
    
    with pytest.raises(KeyError):
        promptline.build(features)
    
    mock_logger.log_error.assert_called_once()
    assert "Error formatting prompt" in mock_logger.log_error.call_args[0][0]
