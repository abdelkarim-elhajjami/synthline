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
            {"id": "gpt-3.5-turbo"},
            {"id": "o3-mini"},
            {"id": "gpt-5"},
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
            {
                "id": "anthropic/claude-sonnet",
                "name": "Claude Sonnet",
                "supported_parameters": [
                    "structured_outputs",
                    "response_format",
                    "max_tokens",
                    "temperature",
                    "top_p",
                ],
            },
            {
                "id": "google/gemini-pro",
                "supported_parameters": ["response_format"],
            },
            {
                "id": "deepseek/deepseek-r1",
                "supported_parameters": [
                    "structured_outputs",
                    "response_format",
                    "max_tokens",
                    "temperature",
                    "top_p",
                ],
            },
            {
                "id": "legacy/no-metadata",
            },
        ]
    }
    mock_client.get = AsyncMock(return_value=mock_response)

    response = client.post(
        "/api/models/fetch",
        json={"provider": "openrouter", "api_key": "sk-or-test"}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    
    assert data[0]["value"] == "openrouter/anthropic/claude-sonnet"
    assert data[0]["label"] == "Claude Sonnet"
    mock_client.get.assert_awaited_once_with(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": "Bearer sk-or-test"},
        params={"supported_parameters": "structured_outputs"},
    )

def test_fetch_models_huggingface_is_not_supported():
    response = client.post(
        "/api/models/fetch",
        json={"provider": "huggingface"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid provider"
