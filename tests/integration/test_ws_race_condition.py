from unittest.mock import patch
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api import app
from dependencies import get_dependencies

client = TestClient(app)


def _mock_deps():
    deps = MagicMock()
    deps.promptline = MagicMock()
    deps.system_ctx = MagicMock()
    deps.logger = MagicMock()
    return deps


def test_websocket_race_condition():
    """
    Validate that optimize endpoint accepts an immediate request
    after WS connection without returning 400.
    """
    connection_id = "test-race-id"
    features = {
        "fm_configuration": {
            "selected_options": {},
            "string_values": {},
            "selected_features": [],
            "or_group_mode": {},
        },
        "samples_per_prompt": 1,
        "pace_iterations": 1,
        "pace_actors": 1,
        "pace_candidates": 1,
        "llm": "openai/gpt-4o-mini",
    }

    deps = _mock_deps()
    app.dependency_overrides[get_dependencies] = lambda: deps
    try:
        with client.websocket_connect(f"/ws/{connection_id}"):
            deps.system_ctx.get_connection.return_value = MagicMock()
            with patch("asyncio.create_task") as mock_create_task:
                def _consume_task(coro):
                    coro.close()
                    return MagicMock()

                mock_create_task.side_effect = _consume_task
                response = client.post(
                    "/api/optimize-prompt",
                    json={"features": features, "connection_id": connection_id},
                )

            assert response.status_code != 400, f"Got 400 Bad Request: {response.text}"
            assert mock_create_task.called
    finally:
        app.dependency_overrides = {}
