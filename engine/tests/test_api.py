from fastapi.testclient import TestClient
from api import app
from dependencies import dependencies

client = TestClient(app)

def test_health_check():
    """Verify that the health check endpoint returns 200 OK."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "engine"}

def test_features_endpoint():
    """Verify that the features endpoint returns a valid structure."""
    response = client.get("/api/features")
    assert response.status_code == 200
    data = response.json()
    assert "ml_task" in data
    assert "artifact" in data
    assert "generator" in data

def test_preview_prompt_validation():
    """Verify validation error when missing required fields."""
    # Sending empty body should trigger validation error
    response = client.post("/api/preview-prompt", json={})
    assert response.status_code == 422

def test_dependencies_singleton():
    """Verify that dependencies are singleton instances."""
    deps1 = dependencies
    deps2 = dependencies
    assert deps1 is deps2
    assert deps1.features is deps1.features
