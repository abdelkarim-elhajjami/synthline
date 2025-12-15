import pytest
from unittest.mock import MagicMock, AsyncMock
from routes.generation import run_generation

@pytest.fixture
def mock_deps():
    deps = MagicMock()
    deps.system_ctx = MagicMock()
    deps.generator = MagicMock()
    deps.output = MagicMock()
    deps.logger = MagicMock()
    
    # Setup Generator to return a sample list
    deps.generator.generate = AsyncMock(return_value=[{"text": "Sample"}])
    deps.generator._fewer_samples_received = False
    
    # Setup Output to return some content
    deps.output.process.return_value = "CSV Content"
    
    return deps

@pytest.mark.asyncio
def test_run_generation_flow(mock_deps):
    """Test full background generation flow."""
    import asyncio
    async def run():
        # Arrange
        connection_id = "test-ws-123"
        features = {"output_format": "CSV", "total_samples": 10}
        
        # Mock WebSocket
        mock_ws = AsyncMock()
        mock_deps.system_ctx.get_connection.return_value = mock_ws
        
        # Act
        await run_generation(features, connection_id, mock_deps)
        
        # Assert
        # 1. Check Generator called
        mock_deps.generator.generate.assert_called_once()
        
        # 2. Check Output Processing
        mock_deps.output.process.assert_called_once()
        
        # 3. Check WebSocket sequence
        # Expect calls: progress(es) -> generation_complete -> complete
        assert mock_ws.send_json.call_count >= 2
        
        # Verify completion message structure
        completion_call = mock_ws.send_json.call_args_list[-2] # Second to last should be results
        args = completion_call[0][0]
        assert args["type"] == "generation_complete"
        assert args["output_content"] == "CSV Content"
        
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
        
        await run_generation({}, "id", mock_deps)
        
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
        
        await run_generation({}, "id", mock_deps)
        
        # Verify error sent to WS
        mock_ws.send_json.assert_called_with({
            "type": "error",
            "message": "Generation error: Gen Failed"
        })

    asyncio.run(run())
