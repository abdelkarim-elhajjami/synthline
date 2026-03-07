"""System context for WebSocket connections."""
from typing import Dict, Optional
from fastapi import WebSocket

class SystemContext:
    """Container for WebSocket connections and current connection state."""
    def __init__(self):
        """Initialize system context."""
        self.connections: Dict[str, WebSocket] = {}
        self.connection_id: Optional[str] = None
        
    def add_connection(self, connection_id: str, websocket: WebSocket) -> None:
        """Register a WebSocket connection."""
        self.connections[connection_id] = websocket
        
    def remove_connection(self, connection_id: str) -> None:
        """Remove a WebSocket connection."""
        self.connections.pop(connection_id, None)
        
    def get_connection(self, connection_id: Optional[str] = None) -> Optional[WebSocket]:
        """Get WebSocket by ID or current connection ID."""
        conn_id = connection_id or self.connection_id
        return self.connections.get(conn_id) if conn_id else None
        
    def set_current_connection(self, connection_id: str) -> None:
        """Set the current connection ID."""
        self.connection_id = connection_id