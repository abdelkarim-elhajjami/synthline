"""
Progress tracking utilities for asynchronous processes.
"""
from typing import Callable, Optional, Awaitable

ProgressFn = Optional[Callable[[float], Awaitable[None]]]
"""Internal 1-param progress callback: async (progress: float) -> None.

The public SDK callback (2-param with message) lives in synthline.types.
"""


async def track_progress(
    callback: ProgressFn,
    progress: float,
) -> None:
    """Invoke callback if provided."""
    if not callback:
        return

    await callback(progress)
