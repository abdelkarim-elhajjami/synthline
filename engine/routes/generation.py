import asyncio
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends

from schemas import GenerationRequest, PromptPreviewRequest, PromptPreviewResponse
from dependencies import get_dependencies, Dependencies

router = APIRouter()

@router.post("/preview-prompt", response_model=PromptPreviewResponse)
async def preview_prompt(
    request: PromptPreviewRequest,
    deps: Dependencies = Depends(get_dependencies)
) -> PromptPreviewResponse:
    """Preview atomic prompts based on the provided configuration."""
    if not deps.promptline:
        raise HTTPException(status_code=503, detail="Promptline service not initialized")
    
    atomic_prompts = deps.promptline.get_atomic_prompts(request.features)
    
    return PromptPreviewResponse(atomic_prompts=atomic_prompts)

@router.post("/generate")
async def start_generation(
    request: GenerationRequest,
    deps: Dependencies = Depends(get_dependencies)
) -> Dict[str, Any]:
    """Generate samples based on the provided configuration."""
    if not deps.generator or not deps.output:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    websocket = deps.system_ctx.get_connection(request.connection_id)
    if not websocket:
        raise HTTPException(status_code=400, detail="WebSocket connection not found")
    
    # Prepare clean features without operational concerns
    features = request.features.copy()
    
    # Start generation in background task
    asyncio.create_task(
        run_generation(
            features,
            request.connection_id,
            deps,
            request.api_keys
        )
    )
    
    return {
        "status": "generation_started", 
        "connection_id": request.connection_id
    }

async def run_generation(
    features: Dict[str, Any],
    connection_id: str,
    deps: Dependencies,
    api_keys: Dict[str, str] = None
) -> None:
    """Run generation in the background and send results via WebSocket."""
    websocket = deps.system_ctx.get_connection(connection_id)
    if not websocket:
        error_message = f"WebSocket connection lost: {connection_id}"
        deps.logger.log_error(error_message, "generation_background")
        return
    
    try:
        # Set the current connection
        deps.system_ctx.set_current_connection(connection_id)
        
        # Progress callback function
        async def progress_callback(progress: float) -> None:
            ws = deps.system_ctx.get_connection(connection_id)
            if ws:
                try:
                    await ws.send_json({"type": "progress", "progress": progress})
                except Exception as ws_e:
                    deps.logger.log_error(f"Failed to send progress: {str(ws_e)}", "websocket")
        
        # Generate samples
        samples = await deps.generator.generate(
            features=features,
            progress_callback=progress_callback,
            api_keys=api_keys
        )
        
        # Process results in memory (stateless)
        output_data = deps.output.process(
            samples=samples, 
            format=features['output_format']
        )
        
        # Send results to client via WebSocket
        websocket = deps.system_ctx.get_connection(connection_id)
        if websocket:
            try:
                # We need to access private member for fewer samples count or expose it. 
                # Ideally, generator returns this info. For now, we access it as before but through deps.
                # A better refactor would be to have generate return a Result object.
                fewer_samples = deps.generator._fewer_samples_received
                
                await websocket.send_json({
                    "type": "generation_complete",
                    "samples": samples,
                    "output_content": output_data, # Send content directly for download
                    "output_format": features['output_format'].lower(),
                    "fewer_samples_received": fewer_samples
                })
                
                # Send completion event
                await websocket.send_json({"type": "complete", "progress": 100})
            except Exception as ws_e:
                deps.logger.log_error(f"Failed to send generation results: {str(ws_e)}", "websocket")
    
    except Exception as e:
        error_message = str(e)
        deps.logger.log_error(error_message, "generation_background", {"connection_id": connection_id})
        
        # Send error to client
        websocket = deps.system_ctx.get_connection(connection_id)
        if websocket:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Generation error: {error_message}"
                })
            except Exception as ws_e:
                deps.logger.log_error(f"Failed to send error: {str(ws_e)}", "websocket")
