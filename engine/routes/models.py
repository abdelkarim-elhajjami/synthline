from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Optional
from pydantic import BaseModel
import httpx

router = APIRouter()

class ModelFetchRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None

@router.post("/fetch")
async def fetch_models(request: ModelFetchRequest):
    """
    Fetch and smartly filter models from OpenAI or OpenRouter.
    """
    if request.provider == "openai":
        if not request.api_key:
             return []
        return await fetch_openai_models(request.api_key)
    elif request.provider == "openrouter":
        return await fetch_openrouter_models(request.api_key)
    else:
        raise HTTPException(status_code=400, detail="Invalid provider")

async def fetch_openai_models(api_key: str) -> List[Dict[str, str]]:
    """
    Fetch OpenAI models and filter for relevant chat models.
    """
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
            ]
            
            return models
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch OpenAI models: {str(e)}")

async def fetch_openrouter_models(api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Fetch all OpenRouter models.
    """
    url = "https://openrouter.ai/api/v1/models"
    # OpenRouter doesn't strictly require auth for the public models endpoint, 
    # but passing key is good practice if they rate limit.
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            models = []
            for item in data.get("data", []):
                mid = item["id"]
                name = item.get("name", mid)
                # Map to our format: value needs 'openrouter/' prefix?
                # In llm.py we use 'openrouter/' prefix to route.
                # BUT OpenRouter IDs already look like "provider/model".
                # Our llm.py checks `startswith('openrouter/')`.
                # So we MUST prepend `openrouter/` to the ID so our backend routes it correctly.
                
                models.append({
                    "value": f"openrouter/{mid}", 
                    "label": name
                })
            
            return models

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch OpenRouter models: {str(e)}")
