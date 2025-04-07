"""
FastAPI routes for the Synthline API.
Provides endpoints for feature retrieval, prompt preview, and data generation.
"""
import os
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from core.fm import FM
from core.generator import Generator
from core.llm import LLMClient
from core.output import Output
from core.promptline import Promptline
from utils.logger import Logger
from utils.ctx import SystemContext

API_TITLE = "Synthline API"
ALLOWED_ORIGINS = ["http://localhost:3000", "http://web:3000"]
LOGS_DIR = "logs"
OUTPUT_DIR = "output"

# Define models
class GenerationRequest(BaseModel):
    features: Dict[str, Any]
    connection_id: str

class GenerationResponse(BaseModel):
    samples: List[Dict[str, Any]]
    output_path: str
    fewer_samples_received: bool = False

class PromptPreviewRequest(BaseModel):
    features: Dict[str, Any]

class PromptPreviewResponse(BaseModel):
    atomic_prompts: Optional[List[Dict[str, Any]]] = None

class OptimizePromptRequest(BaseModel):
    features: Dict[str, Any]
    connection_id: str

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

# Application components
features: Optional[Dict[str, Any]] = None
llm_client: Optional[LLMClient] = None  
promptline: Optional[Promptline] = None
output: Optional[Output] = None
generator: Optional[Generator] = None
logger: Optional[Logger] = None
system_ctx: Optional[SystemContext] = None

@app.on_event("startup")
async def startup_event() -> None:
    """Initialize application components on startup."""
    global features, llm_client, promptline, output, generator, logger, system_ctx
    
    # Get API keys
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not deepseek_key and not openai_key:
        print("Warning: No API keys found in environment variables.")
    
    try:
        logger = Logger(base_dir=LOGS_DIR)
        features = FM().features
        llm_client = LLMClient(logger=logger, deepseek_key=deepseek_key, openai_key=openai_key)
        promptline = Promptline(llm_client=llm_client, logger=logger)
        output = Output(logger=logger, output_dir=OUTPUT_DIR)
        generator = Generator(llm=llm_client, promptline=promptline, logger=logger)
        system_ctx = SystemContext()
        print("Synthline API initialized successfully")
    except Exception as e:
        error_msg = f"Error during startup: {e}"
        print(error_msg)
        logger.log_error(error_msg, "startup")
        raise HTTPException(status_code=500, detail=error_msg)

@app.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str) -> None:
    """Handle WebSocket connections for real-time progress updates."""
    try:
        await websocket.accept()
        system_ctx.add_connection(connection_id, websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            system_ctx.remove_connection(connection_id)
    except Exception as e:
        error_message = f"WebSocket error: {str(e)}"
        logger.log_error(error_message, "websocket", {"connection_id": connection_id})
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/features")
async def get_features() -> Dict[str, Any]:
    """Return all available features and their metadata."""
    if not features:
        raise HTTPException(status_code=503, detail="Features not initialized")
    return features

@app.post("/preview-prompt", response_model=PromptPreviewResponse)
async def preview_prompt(request: PromptPreviewRequest) -> PromptPreviewResponse:
    """Preview atomic prompts based on the provided configuration."""
    if not promptline:
        raise HTTPException(status_code=503, detail="Promptline service not initialized")
    
    atomic_prompts = None
    try:
        atomic_prompts = promptline.get_atomic_prompts(request.features)
    except Exception as e:
        logger.log_error(f"Failed to generate atomic prompts: {str(e)}", "api")
    
    return PromptPreviewResponse(atomic_prompts=atomic_prompts)

@app.post("/generate", response_model=GenerationResponse)
async def generate_samples(request: GenerationRequest) -> GenerationResponse:
    """Generate samples based on the provided configuration."""
    if not generator or not output:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    # Set current connection in system context
    system_ctx.set_current_connection(request.connection_id)
    
    async def progress_callback(progress: float) -> None:
        websocket = system_ctx.get_connection()
        if websocket:
            await websocket.send_json({"type": "progress", "progress": progress})
    
    # Generate samples
    samples = await generator.generate(
        features=request.features,
        progress_callback=progress_callback
    )
    
    # Save and return results
    output_path = output.save(
        samples=samples, 
        format=request.features['output_format'], 
        features=request.features
    )
    
    # Send completion
    websocket = system_ctx.get_connection()
    if websocket:
        await websocket.send_json({"type": "complete", "progress": 100})
    
    return GenerationResponse(
        samples=samples,
        output_path=str(output_path),
        fewer_samples_received=generator._fewer_samples_received
    )

@app.post("/optimize-prompt")
async def start_optimize(request: OptimizePromptRequest) -> Dict[str, Any]:
    """Start optimizing a prompt using PACE."""
    if not promptline:
        raise HTTPException(status_code=503, detail="Promptline service not initialized")
    
    websocket = system_ctx.get_connection(request.connection_id)
    if not websocket:
        raise HTTPException(status_code=400, detail="WebSocket connection not found")
    
    # Prepare clean features without operational concerns
    features = request.features.copy()
    
    try:
        # Start optimization in background task
        asyncio.create_task(
            run_optimization(
                features,
                request.connection_id
            )
        )
        
        return {"status": "optimization_started", "connection_id": request.connection_id}
    except Exception as e:
        error_message = f"Failed to start optimization: {str(e)}"
        logger.log_error(error_message, "optimize_start", {"connection_id": request.connection_id})
        raise HTTPException(status_code=500, detail=error_message)

async def run_optimization(
    features: Dict[str, Any],
    connection_id: str
) -> None:
    """Run optimization in the background and send results via WebSocket."""
    websocket = system_ctx.get_connection(connection_id)
    if not websocket:
        error_message = f"WebSocket connection lost: {connection_id}"
        logger.log_error(error_message, "pace_background")
        return
    
    try:
        # Set the current connection
        system_ctx.set_current_connection(connection_id)
        
        # Progress callback function
        async def progress_callback(progress: float) -> None:
            ws = system_ctx.get_connection()
            if ws:
                try:
                    await ws.send_json({"type": "progress", "progress": progress})
                except Exception as ws_e:
                    logger.log_error(f"Failed to send progress: {str(ws_e)}", "websocket")
        
        # Add connection_id to features for context identification only
        features['connection_id'] = connection_id
        
        # Get atomic configurations
        atomic_configs = promptline.get_atomic_configurations(features)
        
        atomic_prompts = promptline.get_atomic_prompts(features)
        
        # Map the prompts to the configurations
        for i, atomic_config in enumerate(atomic_configs):
            if i < len(atomic_prompts):
                atomic_config['prompt'] = atomic_prompts[i]['prompt']
        
        # Use PACE to optimize all configurations in parallel
        optimized_results = await promptline.optimize_batch(
            atomic_configs=atomic_configs,
            features=features,
            progress_callback=progress_callback,
            system_ctx=system_ctx
        )
        
        websocket = system_ctx.get_connection()
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
                logger.log_error(f"Failed to send optimization batch completion: {str(ws_e)}", "websocket")
    
    except Exception as e:
        error_message = str(e)
        if logger:
            logger.log_error(error_message, "pace_background", {"connection_id": connection_id})
        
        # Send error to client
        websocket = system_ctx.get_connection()
        if websocket:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Optimization error: {error_message}"
                })
            except Exception as ws_e:
                logger.log_error(f"Failed to send error: {str(ws_e)}", "websocket")

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for monitoring and container orchestration."""
    return {"status": "healthy", "service": "engine"}

@app.get("/files/{file_path:path}")
async def serve_file(file_path: str) -> FileResponse:
    """Serve any file from the output directory."""
    try:
        normalized_path = os.path.normpath(file_path)
        if normalized_path.startswith("..") or normalized_path.startswith("/"):
            raise HTTPException(status_code=403, detail="Invalid file path")
            
        full_path = os.path.join(OUTPUT_DIR, normalized_path)
        
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        return FileResponse(
            path=full_path,
            filename=os.path.basename(file_path),
            media_type='application/octet-stream'
        )
    except HTTPException:
        raise
    except Exception as e:
        error_message = f"Error serving file: {str(e)}"
        logger.log_error(error_message, "file_serve", {"file_path": file_path})
        raise HTTPException(status_code=500, detail=error_message)