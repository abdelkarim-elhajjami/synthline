from typing import Any, Dict, Optional
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Depends

from schemas import GenerationRequest, PromptPreviewRequest, PromptPreviewResponse
from dependencies import get_dependencies, Dependencies
from services.generation_service import run_generation
from utils.tasks import create_background_task

router = APIRouter()

@router.post("/preview-prompt", response_model=PromptPreviewResponse)
async def preview_prompt(
    request: PromptPreviewRequest,
    deps: Dependencies = Depends(get_dependencies)
) -> PromptPreviewResponse:
    """Preview atomic prompts based on the provided configuration."""
    try:
        atomic_prompts = deps.promptline.get_atomic_prompts(request.features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    
    return PromptPreviewResponse(atomic_prompts=atomic_prompts)

@router.post("/generate")
async def start_generation(
    request: GenerationRequest,
    deps: Dependencies = Depends(get_dependencies)
) -> Dict[str, Any]:
    """Generate samples based on the provided configuration."""
    try:
        _ = deps.generator
        _ = deps.output_handler
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if not deps.system_ctx.get_connection(request.connection_id):
        raise HTTPException(status_code=400, detail="WebSocket connection not found")
    
    # Prepare clean features without operational concerns
    features = request.features
    align_verify = _validate_generation_features(features)

    operation_id = request.operation_id or str(uuid4())
    if not deps.system_ctx.start_operation(request.connection_id, "generation", operation_id):
        raise HTTPException(
            status_code=409,
            detail="A generation operation is already in progress for this connection.",
        )

    create_background_task(
        run_generation(
            features,
            request.connection_id,
            operation_id,
            deps,
            align_verify=align_verify,
            api_keys=request.api_keys,
        )
    )
    
    return {
        "status": "generation_started", 
        "connection_id": request.connection_id,
        "operation_id": operation_id,
    }


def _validate_generation_features(features: Dict[str, Any]) -> bool:
    """Validate generation features and return the resolved align_verify flag."""
    total_samples = _require_int(features, "total_samples", minimum=1)
    samples_per_prompt = _require_int(features, "samples_per_prompt", minimum=1)
    if samples_per_prompt > total_samples:
        raise HTTPException(
            status_code=422,
            detail="samples_per_prompt cannot be greater than total_samples.",
        )

    align_verify = _resolve_bool(features, "align_verify", default=False)
    if align_verify:
        _require_float(features, "align_threshold", minimum=0.0, maximum=1.0)

    if str(features.get("prompt_approach", "")).upper() == "PACE":
        _require_int(features, "pace_iterations", minimum=1)
        _require_int(features, "pace_actors", minimum=1)
        _require_int(features, "pace_candidates", minimum=1)
        _require_float(features, "pace_alpha", minimum=0.0, maximum=1.0)

    return align_verify


def _require_int(features: Dict[str, Any], key: str, minimum: Optional[int] = None) -> int:
    raw_value = features.get(key)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"'{key}' must be an integer.")

    if minimum is not None and value < minimum:
        raise HTTPException(status_code=422, detail=f"'{key}' must be >= {minimum}.")
    return value


def _require_float(
    features: Dict[str, Any],
    key: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    raw_value = features.get(key)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"'{key}' must be numeric.")

    if minimum is not None and value < minimum:
        raise HTTPException(status_code=422, detail=f"'{key}' must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise HTTPException(status_code=422, detail=f"'{key}' must be <= {maximum}.")
    return value


def _resolve_bool(features: Dict[str, Any], key: str, default: bool = False) -> bool:
    raw_value = features.get(key, default)
    if isinstance(raw_value, bool):
        return raw_value
    raise HTTPException(status_code=422, detail=f"'{key}' must be a boolean.")


