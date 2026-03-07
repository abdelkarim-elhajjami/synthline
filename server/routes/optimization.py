from typing import Dict, Any
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Depends

from schemas import OptimizePromptRequest
from dependencies import get_dependencies, Dependencies
from services.optimization_service import run_optimization
from utils.tasks import create_background_task

router = APIRouter()

@router.post("/optimize-prompt")
async def start_optimize(
    request: OptimizePromptRequest,
    deps: Dependencies = Depends(get_dependencies)
) -> Dict[str, Any]:
    """Start optimizing a prompt using PACE."""
    try:
        _ = deps.promptline
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if not deps.system_ctx.get_connection(request.connection_id):
        raise HTTPException(status_code=400, detail="WebSocket connection not found")

    operation_id = request.operation_id or str(uuid4())
    if not deps.system_ctx.start_operation(request.connection_id, "optimization", operation_id):
        raise HTTPException(
            status_code=409,
            detail="An optimization operation is already in progress for this connection.",
        )

    # Copy because run_optimization augments features with runtime metadata.
    features = request.features.copy()

    create_background_task(
        run_optimization(
            features,
            request.connection_id,
            deps,
            operation_id=operation_id,
            api_keys=request.api_keys,
        )
    )

    return {
        "status": "optimization_started",
        "connection_id": request.connection_id,
        "operation_id": operation_id,
    }
