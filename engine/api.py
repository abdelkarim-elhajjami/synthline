import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from dependencies import dependencies
from routes import features, generation, optimization

API_TITLE = "Synthline API"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the application."""
    # Startup: Initialize container components already done by lazy loading in Container
    # But we can force initialization if needed or log it
    dependencies.logger.log_info("Synthline API starting up...", "startup")
    
    yield
    
    # Shutdown: Clean up resources if needed
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
app.include_router(generation.router, prefix="/api", tags=["generation"])
app.include_router(optimization.router, prefix="/api", tags=["optimization"])

@app.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str) -> None:
    """Handle WebSocket connections for real-time progress updates."""
    try:
        dependencies.system_ctx.add_connection(connection_id, websocket)
        await websocket.accept()
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            dependencies.system_ctx.remove_connection(connection_id)
    except Exception as e:
        error_message = f"WebSocket error: {str(e)}"
        dependencies.logger.log_error(error_message, "websocket", {"connection_id": connection_id})
        # Note: Cannot raise HTTPException in websocket handler effectively after accept, 
        # but we can close with code.
        try:
            await websocket.close(code=1011, reason=error_message)
        except:
            pass

@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "engine"}

# Mount static files (Frontend)
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")