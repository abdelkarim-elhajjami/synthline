"""Background task lifecycle management.

asyncio.create_task() returns a Task that the event loop only holds via a
weak reference.  If no strong reference survives, the garbage collector can
destroy the task mid-execution.  This module keeps a strong reference set so
that fire-and-forget tasks run to completion.

See: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
"""
import asyncio
from typing import Coroutine

_background_tasks: set[asyncio.Task] = set()


def create_background_task(coro: Coroutine) -> asyncio.Task:
    """Schedule *coro* as a background task with a prevent-GC reference."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
