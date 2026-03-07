from typing import Any, Dict, Optional

from fastapi import WebSocket

from utils.ctx import SystemContext
from synthline.utils.logger import Logger


async def send_json_safe(
    websocket: WebSocket,
    payload: Dict[str, Any],
    logger: Logger,
    component: str,
    failure_message: str,
    context: Optional[Dict[str, Any]] = None,
) -> bool:
    """Send a websocket payload and log failures consistently."""
    try:
        await websocket.send_json(payload)
        return True
    except Exception as exc:
        logger.log_error(f"{failure_message}: {str(exc)}", component, context)
        return False


async def send_to_connection(
    system_ctx: SystemContext,
    connection_id: str,
    payload: Dict[str, Any],
    logger: Logger,
    component: str,
    failure_message: str,
    *,
    missing_message: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> bool:
    """Send payload to a connection if it exists."""
    websocket = system_ctx.get_connection(connection_id)
    if not websocket:
        if missing_message:
            logger.log_error(missing_message, component, context)
        return False

    return await send_json_safe(
        websocket=websocket,
        payload=payload,
        logger=logger,
        component=component,
        failure_message=failure_message,
        context=context,
    )
