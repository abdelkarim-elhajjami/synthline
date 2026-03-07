from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Request

from dependencies import get_dependencies, Dependencies
from utils.upload import parse_upload

router = APIRouter()


@router.post("/glossary/upload")
async def upload_glossary(
    request: Request,
    deps: Dependencies = Depends(get_dependencies),
) -> Dict[str, Any]:
    """Upload and activate a glossary YAML at runtime."""
    filename, payload = await parse_upload(request)

    if not (filename.lower().endswith(".yaml") or filename.lower().endswith(".yml")):
        raise HTTPException(status_code=400, detail="Only .yaml and .yml files are supported.")

    try:
        result = deps.update_glossary(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid glossary YAML: {exc}")

    return {
        "status": "uploaded",
        "filename": filename,
        "entries": result.get("entries", 0),
        "replaced": bool(result.get("replaced", False)),
    }
