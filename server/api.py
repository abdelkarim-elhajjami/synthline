import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from settings import settings
from dependencies import dependencies
from routes import features, generation, optimization, models, glossary

API_TITLE = "Synthline API"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the application."""
    dependencies.logger.log_info("Synthline API starting up...", "startup")
    
    yield
    
    # Clean up session data
    sessions_dir = Path.cwd() / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir, ignore_errors=True)
    dependencies.logger.log_info("Synthline API shutting down...", "shutdown")

# Create FastAPI application
app = FastAPI(title=API_TITLE, lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Unexpected error: {str(exc)}"
    dependencies.logger.log_error(error_msg, "global_handler", {"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )

# Include Routers
app.include_router(features.router, prefix="/api", tags=["features"])
app.include_router(glossary.router, prefix="/api", tags=["glossary"])
app.include_router(generation.router, prefix="/api", tags=["generation"])
app.include_router(optimization.router, prefix="/api", tags=["optimization"])
app.include_router(models.router, prefix="/api/models", tags=["models"])

@app.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str) -> None:
    """Handle WebSocket connections for real-time progress updates."""
    try:
        await websocket.accept()
        previous = dependencies.system_ctx.add_connection(connection_id, websocket)
        if previous and previous is not websocket:
            try:
                await previous.close(code=1000, reason="Replaced by newer connection.")
            except Exception:
                pass
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
    except Exception as e:
        error_message = f"WebSocket error: {str(e)}"
        dependencies.logger.log_error(error_message, "websocket", {"connection_id": connection_id})
        try:
            await websocket.close(code=1011, reason=error_message)
        except Exception:
            pass
    finally:
        dependencies.system_ctx.remove_connection(connection_id, websocket=websocket)

@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "engine"}

# Mount static files (Frontend)
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")