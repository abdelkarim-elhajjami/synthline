from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends

from dependencies import get_dependencies, Dependencies

router = APIRouter()

@router.get("/features")
async def get_features(
    deps: Dependencies = Depends(get_dependencies)
) -> Dict[str, Any]:
    """Return all available features and their metadata."""
    if not deps.features:
        raise HTTPException(status_code=503, detail="Features not initialized")
    return deps.features
