"""
Progress tracking utilities for asynchronous processes.
"""
import asyncio
from typing import Callable, Optional, Awaitable

ProgressCallback = Optional[Callable[[float], Awaitable[None]]]

async def track_progress(
    callback: ProgressCallback, 
    progress: float
) -> None:
    """
    Safely call progress callback regardless of its type.
    
    Args:
        callback: The progress callback function, may be async or sync
        progress: Progress value from 0 to 100
    """
    if not callback:
        return
        
    if asyncio.iscoroutinefunction(callback):
        await callback(progress)
    else:
        await asyncio.to_thread(callback, progress) 