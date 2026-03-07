from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request

from dependencies import get_dependencies, Dependencies
from utils.upload import parse_upload

router = APIRouter()

@router.get("/features")
async def get_features(
    deps: Dependencies = Depends(get_dependencies)
) -> Dict[str, Any]:
    """Return all available features and their metadata."""
    try:
        features = deps.features
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if not features:
        raise HTTPException(status_code=503, detail="Features not initialized")
    return features


@router.post("/features/upload")
async def upload_features(
    request: Request,
    deps: Dependencies = Depends(get_dependencies),
) -> Dict[str, Any]:
    """Upload and activate a new feature model XML at runtime."""
    filename, payload = await parse_upload(request)

    if not filename.lower().endswith(".xml"):
        raise HTTPException(status_code=400, detail="Only .xml files are supported.")

    try:
        features = deps.update_feature_model(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FM XML: {exc}")

    return {
        "status": "uploaded",
        "filename": filename,
        "artefact_type": features.get("artefact_type"),
        "source_path": features.get("source_path"),
    }
