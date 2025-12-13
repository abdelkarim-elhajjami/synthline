import pytest
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
    
    # TestClient's websocket_connect is synchronous context manager, 
    # but it simulates the connection.
    # To truly test the race condition, we need to see if the server registers it.
    
    with client.websocket_connect(f"/ws/{connection_id}") as websocket:
        # Immediately try to use the connection
        response = client.post(
            "/api/optimize-prompt",
            json={"features": features, "connection_id": connection_id}
        )
        
        # If the race condition exists, this might be 400
        # If fixed (or not present in TestClient), it's 200/503 (depending on external services)
        
        # Note: We expect 503 because promptline service is mocked/not initiated or 200 if mocked.
        # But definitely NOT 400.
        
        assert response.status_code != 400, f"Got 400 Bad Request: {response.text}"
