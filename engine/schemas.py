from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class GenerationRequest(BaseModel):
    features: Dict[str, Any]
    connection_id: str

class PromptPreviewRequest(BaseModel):
    features: Dict[str, Any]

class PromptPreviewResponse(BaseModel):
    atomic_prompts: Optional[List[Dict[str, Any]]] = None

class OptimizePromptRequest(BaseModel):
    features: Dict[str, Any]
    connection_id: str
