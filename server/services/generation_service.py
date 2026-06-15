"""Thin WebSocket adapter for generation — delegates to SDK."""
from typing import Any, Dict, Optional

from dependencies import Dependencies
from synthline._runtime import runtime_from_deps
from synthline.client import Synthline
from synthline.types import PromptEntry, PromptSet, VerificationEvent
from utils.errors import operation_error_message
from utils.websocket import send_to_connection


async def run_generation(
    features: Dict[str, Any],
    connection_id: str,
    operation_id: str,
    deps: Dependencies,
    *,
    align_verify: bool = False,
    api_keys: Optional[Dict[str, str]] = None,
) -> None:
    """Run generation in the background and send results via WebSocket."""
    if not deps.system_ctx.get_connection(connection_id):
        deps.logger.log_error(
            f"WebSocket connection lost: {connection_id}",
            "generation_background",
            {"connection_id": connection_id, "operation_id": operation_id},
        )
        deps.system_ctx.finish_operation(connection_id, "generation", operation_id)
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

        if features.get("optimized_atomic_prompts"):
            entries = [
                PromptEntry(
                    prompt=p["optimized_prompt"],
                    config=p["config"],
                    score=p.get("pace_score"),
                )
                for p in features["optimized_atomic_prompts"]
            ]
            prompts = PromptSet(
                entries=entries,
                label=str(features.get("classification_label", "")),
                label_definition=str(features.get("classification_label_def", "")),
                samples_per_prompt=int(features.get("samples_per_prompt", 1)),
                optimized=True,
                base_features={
                    "fm_configuration": features.get("fm_configuration"),
                    "classification_label": str(features.get("classification_label", "")),
                    "classification_label_def": str(features.get("classification_label_def", "")),
                    "samples_per_prompt": int(features.get("samples_per_prompt", 1)),
                    "llm": str(features.get("llm", "")),
                    "temperature": float(features.get("temperature", 1.0)),
                    "top_p": float(features.get("top_p", 1.0)),
                    **({"reasoning": features["reasoning"]} if features.get("reasoning") else {}),
                },
            )
        else:
            prompts = sl.build_prompts(
                label=str(features.get("classification_label", "")),
                label_definition=str(features.get("classification_label_def", "")),
                samples_per_prompt=int(features.get("samples_per_prompt", 1)),
                features=features,
                _raw_fm_configuration=True,
            )

            if str(features.get("prompt_approach", "")).upper() == "PACE":
                prompts = await sl.optimize(
                    prompts,
                    alpha=float(features.get("pace_alpha", 0.5)),
                    iterations=int(features.get("pace_iterations", 1)),
                    actors=int(features.get("pace_actors", 4)),
                    candidates=int(features.get("pace_candidates", 2)),
                )

        total_samples = int(features.get("total_samples", 0))
        dataset = await sl.generate(
            prompts,
            samples=total_samples,
            verify=align_verify,
            verify_threshold=float(features.get("align_threshold", 0.5)),
            on_progress=_ws_progress_callback(deps, connection_id, operation_id),
            on_verification=_ws_verification_callback(deps, connection_id, operation_id),
        )

        if _is_active(deps, connection_id, operation_id):
            await send_to_connection(
                system_ctx=deps.system_ctx,
                connection_id=connection_id,
                payload={
                    "type": "generation_complete",
                    "samples": dataset.samples,
                    "output_content": dataset.to_csv(),
                    "metadata": dataset.metadata,
                    "operation": "generation",
                    "operation_id": operation_id,
                },
                logger=deps.logger,
                component="websocket",
                failure_message="Failed to send generation results",
            )

            await send_to_connection(
                system_ctx=deps.system_ctx,
                connection_id=connection_id,
                payload={
                    "type": "complete",
                    "progress": 100,
                    "message": "Generation complete",
                    "operation": "generation",
                    "operation_id": operation_id,
                },
                logger=deps.logger,
                component="websocket",
                failure_message="Failed to send completion event",
            )

    except Exception as exc:
        deps.logger.log_error(
            str(exc),
            "generation_background",
            {"connection_id": connection_id, "operation_id": operation_id},
        )
        await send_to_connection(
            system_ctx=deps.system_ctx,
            connection_id=connection_id,
            payload={
                "type": "error",
                "message": operation_error_message("generation", exc),
                "operation": "generation",
                "operation_id": operation_id,
            },
            logger=deps.logger,
            component="websocket",
            failure_message="Failed to send error",
        )
    finally:
        deps.system_ctx.finish_operation(connection_id, "generation", operation_id)


# ---------------------------------------------------------------------------
# WebSocket callback factories
# ---------------------------------------------------------------------------

def _is_active(deps: Dependencies, connection_id: str, operation_id: str) -> bool:
    return deps.system_ctx.get_active_operation_id(connection_id, "generation") == operation_id


def _ws_progress_callback(deps: Dependencies, connection_id: str, operation_id: str):
    async def callback(progress: float, message: str) -> None:
        if not _is_active(deps, connection_id, operation_id):
            return
        await send_to_connection(
            system_ctx=deps.system_ctx,
            connection_id=connection_id,
            payload={
                "type": "progress",
                "progress": max(0.0, min(progress, 100.0)),
                "message": message,
                "operation": "generation",
                "operation_id": operation_id,
            },
            logger=deps.logger,
            component="websocket",
            failure_message="Failed to send progress",
        )
    return callback


def _ws_verification_callback(deps: Dependencies, connection_id: str, operation_id: str):
    async def callback(event: VerificationEvent) -> None:
        if not _is_active(deps, connection_id, operation_id):
            return
        await send_to_connection(
            system_ctx=deps.system_ctx,
            connection_id=connection_id,
            payload={
                "type": "verification_progress",
                "operation": "generation",
                "operation_id": operation_id,
                "attempt": event.attempt,
                "max_attempts": event.max_attempts,
                "accepted_so_far": event.accepted,
                "samples_needed": event.needed,
                "progress": event.progress,
                "message": "Verifying alignment",
            },
            logger=deps.logger,
            component="websocket",
            failure_message="Failed to send verification progress",
        )
    return callback
