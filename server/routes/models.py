from typing import Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from synthline.core.model_compatibility import is_reasoning_model

router = APIRouter()

OPENROUTER_REQUIRED_PARAMETERS = frozenset({
    "structured_outputs",
    "response_format",
    "max_tokens",
    "temperature",
    "top_p",
})


class ModelFetchRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None


@router.post("/fetch")
async def fetch_models(request: ModelFetchRequest):
    """Fetch provider models, capability-filtering only where metadata allows."""
    if request.provider == "openai":
        if not request.api_key:
            return []
        return await fetch_openai_models(request.api_key)
    if request.provider == "openrouter":
        return await fetch_openrouter_models(request.api_key)
    if request.provider == "huggingface":
        return await fetch_huggingface_models()
    raise HTTPException(status_code=400, detail="Invalid provider")


async def fetch_openai_models(api_key: str) -> List[Dict[str, str]]:
    """Fetch OpenAI models; its catalog does not expose capability metadata."""
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            models = [
                {"value": item["id"], "label": item["id"]}
                for item in data.get("data", [])
                if not is_reasoning_model(item["id"])
            ]

            return models

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch OpenAI models: {str(e)}")


async def fetch_openrouter_models(api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """Fetch standard models that support Synthline's complete request contract."""
    url = "https://openrouter.ai/api/v1/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url,
                headers=headers,
                params={"supported_parameters": "structured_outputs"},
            )
            response.raise_for_status()
            data = response.json()

            models = []
            for item in data.get("data", []):
                supported = set(item.get("supported_parameters") or [])
                mid = item["id"]
                if not OPENROUTER_REQUIRED_PARAMETERS <= supported:
                    continue
                if is_reasoning_model(mid):
                    continue
                models.append({
                    "value": f"openrouter/{mid}",
                    "label": item.get("name", mid),
                })

            return models

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch OpenRouter models: {str(e)}")


async def fetch_huggingface_models() -> List[Dict[str, str]]:
    """Fetch text-generation models; the catalog lacks capability metadata."""
    url = "https://huggingface.co/api/models"
    params = {
        "pipeline_tag": "text-generation",
        "limit": 50,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            models = []
            for item in data:
                model_id = item.get("id", item.get("modelId", ""))
                if model_id and not is_reasoning_model(model_id):
                    models.append({
                        "value": f"huggingface/{model_id}",
                        "label": model_id
                    })

            return models

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch HuggingFace models: {str(e)}")
