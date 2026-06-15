"""Thin WebSocket adapter for optimization — delegates to SDK."""
from typing import Any, Dict, Optional

from dependencies import Dependencies
from synthline._runtime import runtime_from_deps
from synthline.client import Synthline
from synthline.types import PromptUpdateEvent
from utils.errors import operation_error_message
from utils.websocket import send_to_connection


async def run_optimization(
    features: Dict[str, Any],
    connection_id: str,
    deps: Dependencies,
    operation_id: str,
    api_keys: Optional[Dict[str, str]] = None,
) -> None:
    """Run optimization in the background and send results via WebSocket."""
    if not operation_id:
        raise ValueError("operation_id is required for optimization runs.")
    if not deps.system_ctx.get_connection(connection_id):
        deps.logger.log_error(
            f"WebSocket connection lost: {connection_id}",
            "pace_background",
            {"connection_id": connection_id, "operation_id": operation_id},
        )
        deps.system_ctx.finish_operation(connection_id, "optimization", operation_id)
        return

    try:
        runtime = runtime_from_deps(deps)
        sl = Synthline._from_runtime(
            runtime=runtime,
            llm=str(features.get("llm", "")),
            temperature=float(features.get("temperature", 1.0)),
            top_p=float(features.get("top_p", 1.0)),
            reasoning=features.get("reasoning"),
            api_keys=api_keys,
        )

        await send_to_connection(
            system_ctx=deps.system_ctx,
            connection_id=connection_id,
            payload={
                "type": "progress",
                "progress": 0.0,
                "message": "Preparing optimization",
                "operation": "optimization",
                "operation_id": operation_id,
            },
            logger=deps.logger,
            component="websocket",
            failure_message="Failed to send progress",
        )

        # Build prompts (Web UI sends fm_configuration in internal format)
        prompts = sl.build_prompts(
            label=str(features.get("classification_label", "")),
            label_definition=str(features.get("classification_label_def", "")),
            samples_per_prompt=int(features.get("samples_per_prompt", 1)),
            features=features,
            _raw_fm_configuration=True,
        )

        async def progress_callback(progress: float, message: str) -> None:
            await send_to_connection(
                system_ctx=deps.system_ctx,
                connection_id=connection_id,
                payload={
                    "type": "progress",
                    "progress": max(0.0, min(progress, 100.0)),
                    "message": message,
                    "operation": "optimization",
                    "operation_id": operation_id,
                },
                logger=deps.logger,
                component="websocket",
                failure_message="Failed to send progress",
            )

        async def prompt_update_callback(event: PromptUpdateEvent) -> None:
            await send_to_connection(
                system_ctx=deps.system_ctx,
                connection_id=connection_id,
                payload={
                    "type": "prompt_update",
                    "prompt": event.prompt,
                    "score": event.score,
                    "iteration": event.iteration,
                    "atomic_config_index": event.config_index,
                    "total_configs": event.total_configs,
                    "operation": "optimization",
                    "operation_id": operation_id,
                    "message": "Optimizing prompts",
                    "detail": (
                        f"Config {(event.config_index or 0) + 1}/{event.total_configs}, "
                        f"iteration {event.iteration}/{event.total_iterations}, "
                        f"best score {event.score:.3f}"
                    ),
                },
                logger=deps.logger,
                component="websocket",
                failure_message="Failed to send prompt update",
            )

        optimized = await sl.optimize(
            prompts,
            alpha=float(features.get("pace_alpha", 0.5)),
            iterations=int(features.get("pace_iterations", 1)),
            actors=int(features.get("pace_actors", 4)),
            candidates=int(features.get("pace_candidates", 2)),
            on_progress=progress_callback,
            on_prompt_update=prompt_update_callback,
        )

        # Send results in the same format the Web UI expects
        serializable_results = [
            {
                "prompt": entry.prompt,
                "score": float(entry.score) if entry.score is not None else 0.0,
                "atomic_config": entry.config,
            }
            for entry in optimized
        ]

        await send_to_connection(
            system_ctx=deps.system_ctx,
            connection_id=connection_id,
            payload={
                "type": "optimize_complete_batch",
                "optimized_results": serializable_results,
                "operation": "optimization",
                "operation_id": operation_id,
            },
            logger=deps.logger,
            component="websocket",
            failure_message="Failed to send optimization batch completion",
        )

    except Exception as e:
        error_message = str(e)
        deps.logger.log_error(
            error_message,
            "pace_background",
            {"connection_id": connection_id, "operation_id": operation_id},
        )

        await send_to_connection(
            system_ctx=deps.system_ctx,
            connection_id=connection_id,
            payload={
                "type": "error",
                "message": operation_error_message("optimization", e),
                "operation": "optimization",
                "operation_id": operation_id,
            },
            logger=deps.logger,
            component="websocket",
            failure_message="Failed to send error",
        )
    finally:
        deps.system_ctx.finish_operation(connection_id, "optimization", operation_id)
