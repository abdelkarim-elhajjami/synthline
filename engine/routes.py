"""
FastAPI routes for the Synthline API.
Provides endpoints for feature retrieval, prompt preview, and data generation.
"""
import os
from typing import Dict, List, Any, Optional
import asyncio

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fm import FM
from generator import Generator
from llm_client import LLMClient
from output import Output
from promptline import Promptline
from logger import Logger
from pace import PACE

API_TITLE = "Synthline API"
ALLOWED_ORIGINS = ["http://localhost:3000", "http://web:3000"]
LOGS_DIR = "logs"
OUTPUT_DIR = "output"

# Define models
class GenerationRequest(BaseModel):
    feature_values: Dict[str, Any]
    connection_id: str

class GenerationResponse(BaseModel):
    samples: List[Dict[str, Any]]
    output_path: str
    fewer_samples_received: bool = False

class PromptPreviewRequest(BaseModel):
    feature_values: Dict[str, Any]

class PromptPreviewResponse(BaseModel):
    prompt: str

class OptimizePromptRequest(BaseModel):
    feature_values: Dict[str, Any]
    connection_id: str

class OptimizePromptResponse(BaseModel):
    optimized_prompt: str
    score: float

# Create FastAPI application
app = FastAPI(title=API_TITLE)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
connections: Dict[str, WebSocket] = {}

# Application components
features: Optional[Dict[str, Any]] = None
llm_client: Optional[LLMClient] = None  
promptline: Optional[Promptline] = None
output: Optional[Output] = None
generator: Optional[Generator] = None
logger: Optional[Logger] = None

@app.on_event("startup")
async def startup_event() -> None:
    """Initialize application components on startup."""
    global features, llm_client, promptline, output, generator, logger
    
    # Get API keys
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not deepseek_key and not openai_key:
        print("Warning: No API keys found in environment variables.")
    
    try:
        logger = Logger(base_dir=LOGS_DIR)
        features = FM().features
        llm_client = LLMClient(deepseek_key, openai_key, logger=logger)
        promptline = Promptline(logger=logger)
        output = Output(output_dir=OUTPUT_DIR, logger=logger)
        generator = Generator(llm_client, promptline, batch_size=1, logger=logger)
        print("Synthline API initialized successfully")
    except Exception as e:
        print(f"Error during startup: {e}")
        if logger:
            logger.log_error(str(e), "startup")

@app.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str) -> None:
    """Handle WebSocket connections for real-time progress updates."""
    await websocket.accept()
    connections[connection_id] = websocket
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connections.pop(connection_id, None)

@app.get("/features")
async def get_features() -> Dict[str, Any]:
    """Return all available features and their metadata."""
    if not features:
        raise HTTPException(status_code=503, detail="Features not initialized")
    return features

@app.post("/preview-prompt", response_model=PromptPreviewResponse)
async def preview_prompt(request: PromptPreviewRequest) -> PromptPreviewResponse:
    """Generate a prompt preview based on the provided configuration."""
    if not promptline:
        raise HTTPException(status_code=503, detail="Promptline service not initialized")
    
    # Get sample count and build prompt
    count = int(request.feature_values.get('samples_per_prompt'))
    prompt = promptline.build(request.feature_values, count)
    
    return PromptPreviewResponse(prompt=prompt)

@app.post("/generate", response_model=GenerationResponse)
async def generate_samples(request: GenerationRequest) -> GenerationResponse:
    """Generate samples based on the provided configuration."""
    if not generator or not output:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    websocket = connections.get(request.connection_id)
    
    async def progress_callback(progress: float) -> None:
        if websocket:
            await websocket.send_json({"type": "progress", "progress": progress})
    
    # Check if we have an optimized prompt and add it to the request
    optimized_prompt = request.feature_values.get('optimized_prompt')
    
    # Generate samples
    samples = await generator.generate_samples(
        feature_values=request.feature_values,
        progress_callback=progress_callback,
        optimized_prompt=optimized_prompt
    )
    
    # Save and return results
    output_path = output.save(
        samples=samples, 
        format=request.feature_values['output_format'], 
        features=request.feature_values
    )
    
    # Send completion
    if websocket:
        await websocket.send_json({"type": "complete", "progress": 100})
    
    return GenerationResponse(
        samples=samples,
        output_path=str(output_path),
        fewer_samples_received=generator._fewer_samples_received
    )

@app.post("/optimize-prompt")
async def start_optimize(request: OptimizePromptRequest) -> Dict[str, Any]:
    """Start optimizing a prompt using PACE algorithm."""
    if not promptline or not llm_client:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    websocket = connections.get(request.connection_id)
    if not websocket:
        raise HTTPException(status_code=400, detail="WebSocket connection not found")
    
    # Generate initial prompt
    samples_per_prompt = int(request.feature_values.get('samples_per_prompt'))
    initial_prompt = promptline.build(request.feature_values, samples_per_prompt)
    
    # Start optimization in background task
    asyncio.create_task(
        run_optimization(
            request.feature_values,
            initial_prompt,
            request.connection_id
        )
    )
    
    return {"status": "optimization_started", "connection_id": request.connection_id}

async def run_optimization(
    feature_values: Dict[str, Any],
    initial_prompt: str,
    connection_id: str
) -> None:
    """Run optimization in the background and send results via WebSocket."""
    websocket = connections.get(connection_id)
    if not websocket:
        return
    
    try:
        # Progress callback function
        async def progress_callback(progress: float) -> None:
            if websocket:
                await websocket.send_json({"type": "progress", "progress": progress})
        
        # Create PACE optimizer
        pace_optimizer = PACE(
            llm_client=llm_client,
            iterations=int(feature_values.get('pace_iterations')),
            num_actors=int(feature_values.get('pace_actors')),
            initial_prompt=initial_prompt,
            logger=logger,
            connections=connections
        )
        
        feature_values['connection_id'] = connection_id
        
        # Run optimization
        optimized_prompt, score = await pace_optimizer.optimize(
            feature_values=feature_values,
            progress_callback=progress_callback
        )
        
        # Send final result
        if websocket:
            await websocket.send_json({
                "type": "optimize_complete",
                "optimized_prompt": pace_optimizer._clean_prompt(optimized_prompt),
                "score": float(score)
            })
    
    except Exception as e:
        error_message = str(e)
        if logger:
            logger.log_error(error_message, "pace_background", {"connection_id": connection_id})
        
        # Send error to client
        if websocket:
            await websocket.send_json({
                "type": "error",
                "message": f"Optimization error: {error_message}"
            })

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for monitoring and container orchestration."""
    return {"status": "healthy", "service": "engine"}

@app.get("/files/{file_path:path}")
async def serve_file(file_path: str) -> FileResponse:
    """Serve any file from the output directory."""
    full_path = os.path.join(OUTPUT_DIR, file_path)
    
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    return FileResponse(
        path=full_path,
        filename=os.path.basename(file_path),
        media_type='application/octet-stream'
    )

if __name__ == "__main__":
    uvicorn.run("routes:app", host="0.0.0.0", port=8000, reload=True)
