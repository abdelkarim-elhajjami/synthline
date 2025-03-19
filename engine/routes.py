"""
FastAPI routes for the Synthline API.
Provides endpoints for feature retrieval, prompt preview, and data generation.
"""
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator

from fm import FM
from generator import Generator
from llm_client import LLMClient
from output import Output
from promptline import Promptline
from logger import Logger

# Type definitions
T = TypeVar('T')
ProgressCallback = Callable[[float], Awaitable[None]]

# API configuration constants
API_TITLE = "Synthline API"
ALLOWED_ORIGINS = ["http://localhost:3000", "http://web:3000"]
LOGS_DIR = "logs"
OUTPUT_DIR = "output"

# Define models
class GenerationRequest(BaseModel):
    feature_values: Dict[str, Any]
    connection_id: str
    
    @validator('feature_values')
    def validate_feature_values(cls, field_value: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature values for required fields and proper types."""
        required_fields = [
            'domain', 'language', 'label', 'label_definition', 
            'specification_format', 'specification_level', 'stakeholder',
            'total_samples', 'output_format'
        ]
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in field_value]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Convert and validate numeric fields
        field_value['total_samples'] = int(field_value['total_samples'])
        if field_value['total_samples'] <= 0:
            raise ValueError("Total samples must be a positive integer")
        
        if 'samples_per_prompt' in field_value:
            field_value['samples_per_prompt'] = int(field_value['samples_per_prompt'])
            if field_value['samples_per_prompt'] <= 0:
                raise ValueError("Samples per prompt must be a positive integer")
            if field_value['samples_per_prompt'] > field_value['total_samples']:
                raise ValueError("Samples per prompt cannot exceed total samples")
        
        if 'temperature' in field_value:
            field_value['temperature'] = float(field_value['temperature'])
            if not 0 <= field_value['temperature'] <= 2:
                raise ValueError("Temperature must be between 0 and 2")
        
        if 'top_p' in field_value:
            field_value['top_p'] = float(field_value['top_p'])
            if not 0 <= field_value['top_p'] <= 1:
                raise ValueError("Top P must be between 0 and 1")
        
        # Validate output format
        if field_value['output_format'] not in ['JSON', 'CSV']:
            raise ValueError("Output format must be either JSON or CSV")
        
        return field_value

class GenerationResponse(BaseModel):
    samples: List[Dict[str, Any]]
    output_path: str
    fewer_samples_received: bool = False

class PromptPreviewRequest(BaseModel):
    feature_values: Dict[str, Any]
    
    @validator('feature_values')
    def validate_preview_values(cls, field_value: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature values for prompt preview."""
        required_fields = [
            'domain', 'language', 'label', 'label_definition', 
            'specification_format', 'specification_level', 'stakeholder'
        ]
        
        missing_fields = [field for field in required_fields if field not in field_value]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Convert samples_per_prompt to integer if present
        if 'samples_per_prompt' in field_value:
            field_value['samples_per_prompt'] = int(field_value['samples_per_prompt'])
            
        return field_value

class PromptPreviewResponse(BaseModel):
    prompt: str

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
    
    # Get API keys from environment variables
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
            # Keep connection alive until disconnect
            await websocket.receive_text()
    except WebSocketDisconnect:
        if connection_id in connections:
            del connections[connection_id]

@app.get("/features")
async def get_features() -> Dict[str, Any]:
    """Return all available features and their metadata."""
    if not features:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Features not initialized. Service may be starting up."
        )
    return features

@app.post("/preview-prompt", response_model=PromptPreviewResponse)
async def preview_prompt(request: PromptPreviewRequest) -> PromptPreviewResponse:
    """Generate a prompt preview based on the provided configuration."""
    if not promptline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Promptline service not initialized"
        )
    
    try:
        # Build the prompt
        count = request.feature_values.get('samples_per_prompt', 1)
        prompt = promptline.build(request.feature_values, count)
        
        return PromptPreviewResponse(prompt=prompt)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        print(f"Error in preview_prompt: {e}")
        if logger:
            logger.log_error(str(e), "preview_prompt", {"request": str(request)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/generate", response_model=GenerationResponse)
async def generate_samples(request: GenerationRequest) -> GenerationResponse:
    """Generate samples based on the provided configuration."""
    if not generator or not output:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Generator or Output service not initialized"
        )
    
    connection_id = request.connection_id
    websocket = connections.get(connection_id)
    
    try:
        # Define progress callback for WebSocket
        async def progress_callback(progress: float) -> None:
            """Send progress updates through WebSocket if available."""
            if websocket:
                await websocket.send_json({
                    "type": "progress",
                    "progress": progress
                })
        
        # Generate samples with real-time progress
        samples = await generator.generate_samples(
            feature_values=request.feature_values,
            progress_callback=progress_callback
        )
        
        # Save the output
        output_path = output.save(
            samples=samples, 
            format=request.feature_values['output_format'], 
            features=request.feature_values
        )
        
        # Send completion message
        if websocket:
            await websocket.send_json({
                "type": "complete", 
                "progress": 100
            })
        
        # Return the response
        return GenerationResponse(
            samples=samples,
            output_path=str(output_path),
            fewer_samples_received=generator._fewer_samples_received
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        error_msg = f"Generation error: {str(e)}"
        print(error_msg)
        if logger:
            logger.log_error(error_msg, "generate_samples", 
                             {"feature_values": str(request.feature_values)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for monitoring and container orchestration."""
    return {"status": "healthy", "service": "engine"}

@app.get("/download/{llm}/{filename}")
async def download_file(llm: str, filename: str) -> FileResponse:
    """Download a generated file from the output directory."""
    file_path = os.path.join(OUTPUT_DIR, llm, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"File not found: {filename}"
        )
    
    return FileResponse(
        path=file_path, 
        filename=filename,
        media_type='application/octet-stream'
    )

@app.get("/files/{file_path:path}")
async def serve_file(file_path: str) -> FileResponse:
    """Serve a file from the output directory by path."""
    full_path = os.path.join(OUTPUT_DIR, file_path)
    
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"File not found: {file_path}"
        )
    
    # Extract filename for Content-Disposition header
    filename = os.path.basename(file_path)
    
    return FileResponse(
        path=full_path,
        filename=filename,
        media_type='application/octet-stream'
    )

if __name__ == "__main__":
    uvicorn.run("routes:app", host="0.0.0.0", port=8000, reload=True)
