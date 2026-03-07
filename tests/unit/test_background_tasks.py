import pytest
from unittest.mock import MagicMock, AsyncMock
from services.generation_service import run_generation
from services.optimization_service import run_optimization

@pytest.fixture
def mock_deps():
    deps = MagicMock()
    deps.system_ctx = MagicMock()
    deps.generator = MagicMock()
    deps.output_handler = MagicMock()
    deps.promptline = MagicMock()
    deps.logger = MagicMock()
    
    # Setup Generator to return a sample list
    deps.generator.generate = AsyncMock(return_value=[{"text": "Sample", "config": {}}])
    deps.generator._fewer_samples_received = False
    deps.generator.fewer_samples_received = False
    deps.generator.parsing_degraded = False

    # Setup Output to return some content
    deps.output_handler.to_csv.return_value = "CSV Content"
    deps.output_handler.format_sample.return_value = {"Text": "Sample"}
    deps.promptline.get_atomic_prompts.return_value = [{"config": {}, "prompt": "Initial"}]
    deps.promptline.optimize_batch = AsyncMock(return_value=[("Improved", 0.8, {"prompt": "Initial"})])

    active_operations = {}

    def start_operation(connection_id, operation, operation_id):
        connection_ops = active_operations.setdefault(connection_id, {})
        if operation in connection_ops:
            return False
        connection_ops[operation] = operation_id
        return True

    def get_active_operation_id(connection_id, operation):
        return active_operations.get(connection_id, {}).get(operation)

    def finish_operation(connection_id, operation, operation_id=None):
        connection_ops = active_operations.get(connection_id, {})
        active_operation_id = connection_ops.get(operation)
        if active_operation_id is None:
            return
        if operation_id is not None and active_operation_id != operation_id:
            return
        connection_ops.pop(operation, None)
        if not connection_ops and connection_id in active_operations:
            active_operations.pop(connection_id)

    deps.system_ctx.start_operation.side_effect = start_operation
    deps.system_ctx.get_active_operation_id.side_effect = get_active_operation_id
    deps.system_ctx.finish_operation.side_effect = finish_operation
    
    return deps

@pytest.mark.asyncio
def test_run_generation_flow(mock_deps):
    """Test full background generation flow."""
    import asyncio
    async def run():
        # Arrange
        connection_id = "test-ws-123"
        features = {"total_samples": 10}
        
        # Mock WebSocket
        mock_ws = AsyncMock()
        mock_deps.system_ctx.get_connection.return_value = mock_ws

        # Register the operation as active (mimics what the HTTP route does)
        mock_deps.system_ctx.start_operation(connection_id, "generation", "op-1")

        # Act
        await run_generation(features, connection_id, "op-1", mock_deps)
        
        # Assert
        # 1. Check Generator called
        mock_deps.generator.generate.assert_called_once()

        # 2. Check Output formatting
        mock_deps.output_handler.format_sample.assert_called_once()

        # 3. Check WebSocket sequence
        # Expect calls: progress(es) -> generation_complete -> complete
        assert mock_ws.send_json.call_count >= 2

        # Verify completion message structure
        completion_call = mock_ws.send_json.call_args_list[-2] # Second to last should be results
        args = completion_call[0][0]
        assert args["type"] == "generation_complete"
        assert "output_content" in args

        # Verify final complete message
        final_call = mock_ws.send_json.call_args_list[-1]
        assert final_call[0][0]["type"] == "complete"

    asyncio.run(run())

@pytest.mark.asyncio
def test_run_generation_no_ws(mock_deps):
    """Test runs gracefully when WS is missing/disconnected."""
    import asyncio
    async def run():
        mock_deps.system_ctx.get_connection.return_value = None
        
        await run_generation({}, "id", "op-2", mock_deps)
        
        mock_deps.logger.log_error.assert_called()
        # Should NOT attempt generation if WS is gone immediately
        mock_deps.generator.generate.assert_not_called()

    asyncio.run(run())

@pytest.mark.asyncio
def test_run_generation_error_propagation(mock_deps):
    """Test exceptions are sent to client."""
    import asyncio
    async def run():
        mock_ws = AsyncMock()
        mock_deps.system_ctx.get_connection.return_value = mock_ws
        
        # Generator raises error
        mock_deps.generator.generate.side_effect = Exception("Gen Failed")
        
        await run_generation({}, "id", "op-3", mock_deps)
        
        # Verify error sent to WS
        assert mock_ws.send_json.call_count >= 1
        payload = mock_ws.send_json.call_args[0][0]
        assert payload["type"] == "error"
        assert payload["message"] == "Generation error: Gen Failed"
        assert payload["operation"] == "generation"
        assert isinstance(payload.get("operation_id"), str)

    asyncio.run(run())


@pytest.mark.asyncio
def test_run_generation_llm_error_payload(mock_deps):
    """Test typed LLM errors are sent with machine-readable fields."""
    import asyncio

    async def run():
        mock_ws = AsyncMock()
        mock_deps.system_ctx.get_connection.return_value = mock_ws
        mock_deps.generator.generate.side_effect = Exception("LLM rate-limited: 429 Too Many Requests")

        await run_generation({}, "id", "op-4", mock_deps)

        assert mock_ws.send_json.call_count >= 1
        payload = mock_ws.send_json.call_args[0][0]
        assert payload["type"] == "error"
        assert payload["message"] == "Generation error: LLM rate-limited: 429 Too Many Requests"
        assert payload["operation"] == "generation"
        assert isinstance(payload.get("operation_id"), str)

    asyncio.run(run())


@pytest.mark.asyncio
def test_run_optimization_llm_error_payload(mock_deps):
    """Test optimization reports typed LLM errors over websocket."""
    import asyncio

    async def run():
        mock_ws = AsyncMock()
        mock_deps.system_ctx.get_connection.return_value = mock_ws
        mock_deps.promptline.optimize_batch.side_effect = Exception(
            "LLM rate-limited: 429 Too Many Requests"
        )

        features = {
            "fm_configuration": {},
            "pace_iterations": 1,
            "pace_actors": 1,
            "pace_candidates": 1,
        }
        await run_optimization(features, "id", mock_deps, operation_id="op-5")

        assert mock_ws.send_json.call_count >= 1
        payload = mock_ws.send_json.call_args[0][0]
        assert payload["type"] == "error"
        assert payload["message"] == "Optimization error: LLM rate-limited: 429 Too Many Requests"
        assert payload["operation"] == "optimization"
        assert isinstance(payload.get("operation_id"), str)

    asyncio.run(run())
