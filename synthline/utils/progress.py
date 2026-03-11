"""
Progress tracking utilities for asynchronous processes.
"""
from synthline.types import ProgressCallback


async def report_progress(
    callback: ProgressCallback,
    progress: float,
    message: str,
) -> None:
    """Invoke callback if provided."""
    if callback:
        await callback(progress, message)
