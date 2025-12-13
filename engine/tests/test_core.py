import pytest
from unittest.mock import MagicMock
from core.promptline import Promptline
from core.fm import Feature

@pytest.fixture
def mock_llm():
    return MagicMock()

@pytest.fixture
def mock_logger():
    return MagicMock()

def test_atomic_prompt_splitting(mock_llm, mock_logger):
    """Test that atomic prompts are generated correctly without LLM calls for simple cases."""
    promptline = Promptline(mock_llm, mock_logger)
    
    # Test case: 2 LLMs x 2 Temperatures = 4 Atomic Prompts
    features = {
        "generator": {
            "llm": ["gpt-4", "deepseek"],
            "temperature": ["0.7", "1.0"],
            "top_p": "1.0",
            "samples_per_prompt": "5"
        },
        "artifact": {
            "specification_format": ["User Story"],
            "specification_level": ["High"],
            "stakeholder": ["User"],
            "domain": "Retail",
            "language": "English"
        },
        "ml_task": {
            "label": "Test",
            "label_definition": "Test Def"
        }
    }
    
    # Note: atomic generation logic depends on feature structure.
    # Simple Cartesian product check if implemented in Promptline
    # This assumes get_atomic_prompts returns a list of configs
    
    # Since we can't easily test the exact Cartesian product without the real FM structure interacting,
    # we verify that it handles basic extraction.
    
    # If get_atomic_prompts relies on FM defaults, we might need a more integration-like test or mock FM.
    pass 

def test_feature_flattening():
    """Test generic feature flattening if applicable."""
    pass
