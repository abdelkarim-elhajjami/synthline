"""System context for WebSocket connections and in-flight operations."""
from typing import Dict, Optional
from fastapi import WebSocket

class SystemContext:
    """Container for active WebSocket connections."""
    def __init__(self):
        """Initialize system context."""
        self.connections: Dict[str, WebSocket] = {}
        self.active_operations: Dict[str, Dict[str, str]] = {}
        
    def add_connection(self, connection_id: str, websocket: WebSocket) -> Optional[WebSocket]:
        """Register a WebSocket connection and return replaced connection if present."""
        previous = self.connections.get(connection_id)
        self.connections[connection_id] = websocket
        return previous
        
    def remove_connection(self, connection_id: str, websocket: Optional[WebSocket] = None) -> None:
        """
        Remove a WebSocket connection.

        When websocket is provided, remove only if it is the current connection for the ID.
        """
        current = self.connections.get(connection_id)
        if websocket is not None and current is not websocket:
            return
        self.connections.pop(connection_id, None)
        self.active_operations.pop(connection_id, None)
        
    def get_connection(self, connection_id: str) -> Optional[WebSocket]:
        """Get WebSocket by connection ID."""
        return self.connections.get(connection_id)

    def start_operation(self, connection_id: str, operation: str, operation_id: str) -> bool:
        """Start an operation if no in-flight operation of the same type exists."""
        connection_operations = self.active_operations.setdefault(connection_id, {})
        if operation in connection_operations:
            return False
        connection_operations[operation] = operation_id
        return True

    def get_active_operation_id(self, connection_id: str, operation: str) -> Optional[str]:
        """Return active operation id for a connection + operation type."""
        return self.active_operations.get(connection_id, {}).get(operation)

    def finish_operation(
        self,
        connection_id: str,
        operation: str,
        operation_id: Optional[str] = None,
    ) -> None:
        """Finish an operation, optionally requiring operation_id match."""
        connection_operations = self.active_operations.get(connection_id)
        if not connection_operations:
            return
        active_operation_id = connection_operations.get(operation)
        if active_operation_id is None:
            return
        if operation_id is not None and active_operation_id != operation_id:
            return
        connection_operations.pop(operation, None)
        if not connection_operations:
            self.active_operations.pop(connection_id, None)