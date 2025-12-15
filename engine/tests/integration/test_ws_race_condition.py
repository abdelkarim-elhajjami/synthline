from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_websocket_race_condition():
    """
    Test that sending a request immediately after WS connection works.
    This mimics the frontend behavior.
    """
    connection_id = "test-race-id"
    features = {
        "generator": {"llm": "gpt-4"}, 
        "artifact": {}, 
        "ml_task": {}
    }
    
    with client.websocket_connect(f"/ws/{connection_id}") as websocket:
        response = client.post(
            "/api/optimize-prompt",
            json={"features": features, "connection_id": connection_id}
        )
        
        # This test aims to prevent a race condition that could result in a 400 Bad Request.
        # A successful response should yield a 200 OK or 503 Service Unavailable (if external services are mocked/not running).
        # The critical check is that the response status code must not be 400.
        assert response.status_code != 400, f"Got 400 Bad Request: {response.text}"
