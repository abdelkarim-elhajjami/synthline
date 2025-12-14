import asyncio
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, WebSocket

from schemas import OptimizePromptRequest
from dependencies import get_dependencies, Dependencies

router = APIRouter()

@router.post("/optimize-prompt")
async def start_optimize(
    request: OptimizePromptRequest,
    deps: Dependencies = Depends(get_dependencies)
) -> Dict[str, Any]:
    """Start optimizing a prompt using PACE."""
    if not deps.promptline:
        raise HTTPException(status_code=503, detail="Promptline service not initialized")
    
    websocket = deps.system_ctx.get_connection(request.connection_id)
    if not websocket:
        raise HTTPException(status_code=400, detail="WebSocket connection not found")
    
    # Prepare clean features without operational concerns
    features = request.features.copy()
    
    # Start optimization in background task
    asyncio.create_task(
        run_optimization(
            features,
            request.connection_id,
            deps,
            request.api_keys
        )
    )
    
    return {"status": "optimization_started", "connection_id": request.connection_id}

async def run_optimization(
    features: Dict[str, Any],
    connection_id: str,
    deps: Dependencies,
    api_keys: Dict[str, str] = None
) -> None:
    """Run optimization in the background and send results via WebSocket."""
    websocket = deps.system_ctx.get_connection(connection_id)
    if not websocket:
        error_message = f"WebSocket connection lost: {connection_id}"
        deps.logger.log_error(error_message, "pace_background")
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
        
        # Add connection_id to features for context identification only
        features['connection_id'] = connection_id
        
        # Get atomic configurations
        atomic_configs = deps.promptline.get_atomic_configurations(features)
        
        atomic_prompts = deps.promptline.get_atomic_prompts(features)
        
        # Map the prompts to the configurations
        for i, atomic_config in enumerate(atomic_configs):
            if i < len(atomic_prompts):
                atomic_config['prompt'] = atomic_prompts[i]['prompt']
        
        # Use PACE to optimize all configurations in parallel
        optimized_results = await deps.promptline.optimize_batch(
            atomic_configs=atomic_configs,
            features=features,
            progress_callback=progress_callback,
            system_ctx=deps.system_ctx,
            api_keys=api_keys
        )
        
        websocket = deps.system_ctx.get_connection(connection_id)
        if websocket:
            try:
                # Filter out non-serializable objects from configs
                serializable_results = []
                for prompt, score, atomic_config in optimized_results:
                    clean_atomic_config = {}
                    for k, v in atomic_config.items():
                        # Exclude operational system context properties
                        if k != 'connections' and not isinstance(v, WebSocket):
                            clean_atomic_config[k] = v
                    
                    serializable_results.append({
                        "prompt": prompt,
                        "score": float(score),
                        "atomic_config": clean_atomic_config
                    })
                
                await websocket.send_json({
                    "type": "optimize_complete_batch",
                    "optimized_results": serializable_results
                })
            except Exception as ws_e:
                deps.logger.log_error(f"Failed to send optimization batch completion: {str(ws_e)}", "websocket")
    
    except Exception as e:
        error_message = str(e)
        deps.logger.log_error(error_message, "pace_background", {"connection_id": connection_id})
        
        # Send error to client
        websocket = deps.system_ctx.get_connection(connection_id)
        if websocket:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Optimization error: {error_message}"
                })
            except Exception as ws_e:
                deps.logger.log_error(f"Failed to send error: {str(ws_e)}", "websocket")
