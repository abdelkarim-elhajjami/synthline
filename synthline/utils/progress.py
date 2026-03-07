"""
Progress tracking utilities for asynchronous processes.
"""
from typing import Callable, Optional, Awaitable

ProgressCallback = Optional[Callable[[float], Awaitable[None]]]

async def track_progress(
    callback: ProgressCallback,
    progress: float
) -> None:
    """
    Call the async progress callback if provided.

    Args:
        callback: The async progress callback function
        progress: Progress value from 0 to 100
    """
    if not callback:
        return

    await callback(progress)
