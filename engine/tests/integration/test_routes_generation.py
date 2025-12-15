import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from api import app
from dependencies import get_dependencies

client = TestClient(app)

@pytest.fixture
def mock_deps():
    """Create a mock for dependencies."""
    deps = MagicMock()
    deps.promptline = MagicMock()
    deps.generator = MagicMock()
    deps.output = MagicMock()
    deps.system_ctx = MagicMock()
    deps.logger = MagicMock()
    
    return deps

@pytest.fixture(autouse=True)
def override_deps(mock_deps):
    """Override the global dependency with our mock."""
    app.dependency_overrides[get_dependencies] = lambda: mock_deps
    yield
    app.dependency_overrides = {}

def test_preview_prompt_success(mock_deps):
    """Test generating atomic prompt previews."""
    # Arrange
    features = {
        "generator": {"llm": "gpt-4"}, 
        "samples_per_prompt": 1,
    }
    expected_prompts = [
        {"config": features, "prompt": "Generated Prompt"}
    ]
    mock_deps.promptline.get_atomic_prompts.return_value = expected_prompts
    
    # Act
    response = client.post("/api/preview-prompt", json={"features": features})
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["atomic_prompts"] == expected_prompts
    mock_deps.promptline.get_atomic_prompts.assert_called_once()

def test_generate_start_success(mock_deps):
    """Test starting a generation task."""
    # Arrange
    connection_id = "test-conn-123"
    features = {"generator": {"llm": "gpt-4"}, "output": {"output_format": "JSON"}}
    
    # Mock WebSocket connection existence
    mock_socket = MagicMock()
    mock_deps.system_ctx.get_connection.return_value = mock_socket
    
    # Act
    with patch("asyncio.create_task") as mock_create_task:
        response = client.post(
            "/api/generate", 
            json={"features": features, "connection_id": connection_id}
        )
        
        # Assert
        assert response.status_code == 200
        assert response.json()["status"] == "generation_started"
        assert response.json()["connection_id"] == connection_id
        
        # Verify background task was scheduled
        mock_create_task.assert_called_once()

def test_generate_missing_websocket(mock_deps):
    """Test that generation fails if WebSocket isn't connected."""
    # Arrange
    mock_deps.system_ctx.get_connection.return_value = None
    
    # Act
    response = client.post(
        "/api/generate", 
        json={"features": {}, "connection_id": "missing-id"}
    )
    
    # Assert
    assert response.status_code == 400
    assert "WebSocket connection not found" in response.json()["detail"]

def test_preview_prompt_validation_error():
    """Test Pydantic validation for missing fields."""
    # Act
    response = client.post("/api/preview-prompt", json={}) # Missing 'features'
    
    # Assert
    assert response.status_code == 422

def test_optimize_prompt_start_success(mock_deps):
    """Test starting an optimization task."""
    # Arrange
    connection_id = "test-opt-conn-123"
    features = {"pace_iterations": 2, "samples_per_prompt": 1}
    
    # Mock WebSocket existence
    mock_socket = MagicMock()
    mock_deps.system_ctx.get_connection.return_value = mock_socket
    
    # Act
    with patch("asyncio.create_task") as mock_create_task:
        response = client.post(
            "/api/optimize-prompt",
            json={
                "features": features, 
                "connection_id": connection_id,
                "api_keys": {}
            }
        )
        
        # Assert
        assert response.status_code == 200
        assert response.json()["status"] == "optimization_started"
        
        # Verify background task scheduled
        mock_create_task.assert_called_once()

def test_optimize_prompt_no_ws(mock_deps):
    """Test optimization failure when WebSocket is missing."""
    # Arrange
    mock_deps.system_ctx.get_connection.return_value = None
    
    # Act
    response = client.post(
        "/api/optimize-prompt",
        json={
            "features": {}, 
            "connection_id": "missing-id",
            "api_keys": {}
        }
    )
    
    # Assert
    assert response.status_code == 400
    assert "WebSocket connection not found" in response.json()["detail"]
