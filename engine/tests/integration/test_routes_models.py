from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from api import app

client = TestClient(app)

def test_fetch_models_invalid_provider():
    """Test validation for invalid provider."""
    response = client.post(
        "/api/models/fetch",
        json={"provider": "invalid-provider", "api_key": "dummy"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid provider"

@patch("httpx.AsyncClient")
def test_fetch_models_openai_success(mock_client_cls):
    """Test fetching OpenAI models successfully."""
    mock_client = MagicMock()
    mock_client_cls.return_value.__aenter__.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "gpt-4"},
            {"id": "gpt-3.5-turbo"}
        ]
    }
    mock_client.get = AsyncMock(return_value=mock_response)

    response = client.post(
        "/api/models/fetch",
        json={"provider": "openai", "api_key": "sk-test"}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["value"] == "gpt-4"
    assert data[1]["value"] == "gpt-3.5-turbo"

@patch("httpx.AsyncClient")
def test_fetch_models_openrouter_success(mock_client_cls):
    """Test fetching OpenRouter models successfully."""
    mock_client = MagicMock()
    mock_client_cls.return_value.__aenter__.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus"},
            {"id": "google/gemini-pro"} # No name provided
        ]
    }
    mock_client.get = AsyncMock(return_value=mock_response)

    response = client.post(
        "/api/models/fetch",
        json={"provider": "openrouter", "api_key": "sk-or-test"}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    
    assert data[0]["value"] == "openrouter/anthropic/claude-3-opus"
    assert data[0]["label"] == "Claude 3 Opus"
    
    assert data[1]["value"] == "openrouter/google/gemini-pro"
    assert data[1]["label"] == "google/gemini-pro" # Fallback to ID if name missing
