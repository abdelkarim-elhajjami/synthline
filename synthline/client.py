"""Synthline SDK client — the public entry point."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from synthline.core.align_verifier import AlignVerifier
from synthline.core.constants import OPERATING_FIELDS
from synthline.core.fm_parser import FM

from synthline._runtime import Runtime, create_runtime, runtime_from_deps
from synthline.types import Dataset, PromptEntry, PromptSet


# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------

ProgressCallback = Optional[Callable[[float, str], Awaitable[None]]]
"""async (progress_pct: float, message: str) -> None"""

VerificationCallback = Optional[
    Callable[[int, int, int, int, float], Awaitable[None]]
]
"""async (attempt, max_attempts, accepted_so_far, samples_needed, progress_pct) -> None"""

PromptUpdateCallback = Optional[
    Callable[[str, float, int, int, int, int], Awaitable[None]]
]
"""async (prompt, score, iteration, n_iterations, config_index, total_configs) -> None"""


# ---------------------------------------------------------------------------
# Synthline
# ---------------------------------------------------------------------------

class Synthline:
    """Synthline SDK — build prompts, optimize, and generate synthetic data.

    Parameters
    ----------
    fm : str
        Path to a FeatureIDE ``fm.xml`` file.
    glossary : str | None
        Path to a YAML glossary file.
    llm : str
        LLM identifier (e.g. ``"openrouter/meta-llama/llama-3.3-70b-instruct"``).
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling probability.
    api_keys : dict | None
        Provider API keys (``{"openrouter": "sk-..."}``).  Falls back to
        environment variables when omitted.
    debug : bool
        Enable debug logging.
    """

    def __init__(
        self,
        fm: str,
        llm: str,
        *,
        glossary: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        api_keys: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ) -> None:
        self._runtime = create_runtime(
            fm_path=fm,
            glossary_path=glossary,
            api_keys=api_keys,
            debug=debug,
        )
        self._llm = llm
        self._temperature = temperature
        self._top_p = top_p
        self._api_keys = api_keys

    # -- convenience properties ---------------------------------------------

    @property
    def fm(self) -> FM:
        """The parsed Feature Model."""
        return self._runtime.fm

    # -- alternate constructor (internal) ------------------------------------

    @classmethod
    def _from_runtime(
        cls,
        runtime: Runtime,
        llm: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        api_keys: Optional[Dict[str, str]] = None,
    ) -> "Synthline":
        """Build a Synthline instance that reuses an existing Runtime.

        This is used by Web UI service adapters to avoid re-initializing
        heavy objects that already live in the DI container.
        """
        instance = cls.__new__(cls)
        instance._runtime = runtime
        instance._llm = llm
        instance._temperature = temperature
        instance._top_p = top_p
        instance._api_keys = api_keys
        return instance

    # ======================================================================
    # build_prompts
    # ======================================================================

    def build_prompts(
        self,
        label: str,
        label_definition: str,
        samples_per_prompt: int,
        features: Dict[str, List[str]],
        *,
        or_group_mode: Optional[Dict[str, str]] = None,
        _raw_fm_configuration: bool = False,
    ) -> PromptSet:
        """Build atomic prompts from a feature selection.  No LLM call."""
        return asyncio.run(
            self.abuild_prompts(
                label=label,
                label_definition=label_definition,
                samples_per_prompt=samples_per_prompt,
                features=features,
                or_group_mode=or_group_mode,
                _raw_fm_configuration=_raw_fm_configuration,
            )
        )

    async def abuild_prompts(
        self,
        label: str,
        label_definition: str,
        samples_per_prompt: int,
        features: Dict[str, Any],
        *,
        or_group_mode: Optional[Dict[str, str]] = None,
        _raw_fm_configuration: bool = False,
    ) -> PromptSet:
        """Async version of :meth:`build_prompts`."""
        if _raw_fm_configuration:
            # Web UI path: features dict already contains fm_configuration
            internal = dict(features)
        else:
            fm_configuration = _translate_features(self._runtime.fm, features, or_group_mode)
            internal = {
                "fm_configuration": fm_configuration,
                "classification_label": label,
                "classification_label_def": label_definition,
                "samples_per_prompt": samples_per_prompt,
                "llm": self._llm,
                "temperature": self._temperature,
                "top_p": self._top_p,
            }

        atomic_prompts = self._runtime.promptline.get_atomic_prompts(internal)

        entries = [
            PromptEntry(prompt=ap["prompt"], config=ap["config"])
            for ap in atomic_prompts
        ]
        return PromptSet(
            entries=entries,
            label=label,
            label_definition=label_definition,
            samples_per_prompt=samples_per_prompt,
            optimized=False,
            _features=internal,
        )

    # ======================================================================
    # optimize
    # ======================================================================

    def optimize(
        self,
        prompts: PromptSet,
        *,
        alpha: float = 0.5,
        iterations: int = 1,
        actors: int = 4,
        candidates: int = 2,
        on_progress: ProgressCallback = None,
        on_prompt_update: PromptUpdateCallback = None,
    ) -> PromptSet:
        """Optimize prompts via PACE.  Returns a new PromptSet with updated text + scores."""
        return asyncio.run(
            self.aoptimize(
                prompts,
                alpha=alpha,
                iterations=iterations,
                actors=actors,
                candidates=candidates,
                on_progress=on_progress,
                on_prompt_update=on_prompt_update,
            )
        )

    async def aoptimize(
        self,
        prompts: PromptSet,
        *,
        alpha: float = 0.5,
        iterations: int = 1,
        actors: int = 4,
        candidates: int = 2,
        on_progress: ProgressCallback = None,
        on_prompt_update: PromptUpdateCallback = None,
    ) -> PromptSet:
        """Async version of :meth:`optimize`."""
        # Prepare atomic configs (same as optimization_service does)
        atomic_configs: List[Dict[str, Any]] = []
        for entry in prompts.entries:
            config = dict(entry.config)
            config["prompt"] = entry.prompt
            atomic_configs.append(config)

        features: Dict[str, Any] = {
            **prompts._features,
            "llm": self._llm,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "pace_alpha": alpha,
            "pace_iterations": iterations,
            "pace_actors": actors,
            "pace_candidates": candidates,
            "prompt_approach": "PACE",
        }

        # Wrap SDK callbacks into the format Promptline.optimize_batch expects
        _progress_cb = None
        if on_progress:
            async def _progress_cb(progress: float) -> None:
                await on_progress(progress, "Optimizing prompts")

        _prompt_update_cb = None
        if on_prompt_update:
            _prompt_update_cb = on_prompt_update

        optimized_results = await self._runtime.promptline.optimize_batch(
            atomic_configs=atomic_configs,
            features=features,
            progress_callback=_progress_cb,
            prompt_update_callback=_prompt_update_cb,
            api_keys=self._api_keys,
        )

        # Build new PromptSet with optimized prompts and scores
        entries = [
            PromptEntry(
                prompt=optimized_prompt,
                config=atomic_config,
                score=score,
            )
            for optimized_prompt, score, atomic_config in optimized_results
        ]

        return PromptSet(
            entries=entries,
            label=prompts.label,
            label_definition=prompts.label_definition,
            samples_per_prompt=prompts.samples_per_prompt,
            optimized=True,
            _features=prompts._features,
        )

    # ======================================================================
    # generate
    # ======================================================================

    def generate(
        self,
        prompts: PromptSet,
        samples: int,
        *,
        verify: bool = False,
        verify_threshold: float = 0.5,
        on_progress: ProgressCallback = None,
        on_verification: VerificationCallback = None,
    ) -> Dataset:
        """Generate synthetic data from a PromptSet."""
        return asyncio.run(
            self.agenerate(
                prompts,
                samples,
                verify=verify,
                verify_threshold=verify_threshold,
                on_progress=on_progress,
                on_verification=on_verification,
            )
        )

    async def agenerate(
        self,
        prompts: PromptSet,
        samples: int,
        *,
        verify: bool = False,
        verify_threshold: float = 0.5,
        on_progress: ProgressCallback = None,
        on_verification: VerificationCallback = None,
    ) -> Dataset:
        """Async version of :meth:`generate`."""
        started = time.perf_counter()
        run_id = str(uuid4())

        features = prompts.to_features_dict(
            total_samples=samples,
            llm=self._llm,
            temperature=self._temperature,
            top_p=self._top_p,
            api_keys=self._api_keys,
        )

        # --- progress adapter --------------------------------------------------
        gen_progress_cb = None
        if on_progress:
            async def gen_progress_cb(progress: float) -> None:
                displayed = progress * 0.8 if verify else progress
                await on_progress(displayed, "Generating samples")

        # --- generation --------------------------------------------------------
        raw_samples = await self._runtime.generator.generate(
            features=features,
            progress_callback=gen_progress_cb,
            api_keys=self._api_keys,
        )

        # --- verification (optional) -------------------------------------------
        warnings: List[str] = []
        verification_block: Any = False

        if verify and raw_samples:
            if on_progress:
                await on_progress(80.0, "Verifying alignment")

            raw_samples, verification_block, vf_warnings = await self._run_verification_loop(
                raw_samples=raw_samples,
                features=features,
                threshold=verify_threshold,
                samples_needed=samples,
                on_verification=on_verification,
            )
            warnings.extend(vf_warnings)
        else:
            deficit = max(samples - len(raw_samples), 0)
            if deficit > 0:
                warnings.append("generation_deficit")

        # --- format samples ----------------------------------------------------
        formatted = [
            self._runtime.output_handler.format_sample(s["text"], s["config"])
            for s in raw_samples
        ]

        # --- report ------------------------------------------------------------
        report = _build_report(
            run_id=run_id,
            features=features,
            raw_samples=raw_samples,
            requested=samples,
            produced=len(raw_samples),
            warnings=warnings,
            verification_block=verification_block,
        )

        # --- metadata ----------------------------------------------------------
        metadata: Dict[str, Any] = {
            "llm": self._llm,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "samples_requested": samples,
            "samples_produced": len(formatted),
            "verify": verify,
            "verify_threshold": verify_threshold if verify else None,
            "optimized": prompts.optimized,
            "duration_seconds": round(time.perf_counter() - started, 2),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        if on_progress:
            await on_progress(100.0, "Complete")

        return Dataset(samples=formatted, report=report, metadata=metadata)

    # -- verification loop (extracted from generation_service) ---------------

    async def _run_verification_loop(
        self,
        raw_samples: List[Dict[str, Any]],
        features: Dict[str, Any],
        threshold: float,
        samples_needed: int,
        on_verification: VerificationCallback = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
        """Verify samples, regenerating rejected ones up to MAX_RETRIES."""
        verifier = self._runtime.align_verifier
        all_accepted: List[Dict[str, Any]] = []
        accepted_scores: List[float] = []
        rejected_scores: List[float] = []
        accepted_per_attempt: List[int] = []
        attempt_trace: List[Dict[str, Any]] = []
        warnings: List[str] = []
        total_generated = len(raw_samples)
        termination_reason = "max_retries_reached"
        attempts_used = 0
        pending = raw_samples
        total_attempts = AlignVerifier.MAX_RETRIES + 1

        for attempt in range(total_attempts):
            attempts_used = attempt + 1
            started = time.perf_counter()

            if on_verification:
                progress = 80.0 + ((attempt / total_attempts) * 19.0)
                await on_verification(
                    attempt + 1, total_attempts,
                    len(all_accepted), samples_needed, progress,
                )

            accepted, rejected = verifier.verify(pending, threshold)
            rejected_scores.extend(score for _, score in rejected)

            slots_remaining = samples_needed - len(all_accepted)
            added = 0
            for sample, score in accepted[:slots_remaining]:
                all_accepted.append(sample)
                accepted_scores.append(score)
                added += 1

            accepted_per_attempt.append(added)

            if on_verification:
                ratio = len(all_accepted) / samples_needed if samples_needed > 0 else 1.0
                progress = 80.0 + (((attempt + ratio) / total_attempts) * 19.0)
                await on_verification(
                    attempt + 1, total_attempts,
                    len(all_accepted), samples_needed, progress,
                )

            attempt_trace.append({
                "attempt": attempt + 1,
                "pending_in": len(pending),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "accepted_added": added,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            })

            if len(all_accepted) >= samples_needed:
                termination_reason = "quota_met"
                break
            if not rejected:
                termination_reason = "no_rejected_remaining"
                break
            if attempt == AlignVerifier.MAX_RETRIES:
                break

            # Regenerate rejected portion
            deficit = samples_needed - len(all_accepted)
            regen_features = dict(features)
            regen_features["total_samples"] = deficit
            pending = await self._runtime.generator.generate(
                features=regen_features,
                progress_callback=None,
                api_keys=self._api_keys,
            )
            total_generated += len(pending)

            if self._runtime.generator.fewer_samples_received and "fewer_samples_received" not in warnings:
                warnings.append("fewer_samples_received")
            if self._runtime.generator.parsing_degraded and "parsing_degraded" not in warnings:
                warnings.append("parsing_degraded")
            if not pending:
                termination_reason = "generation_returned_empty"
                break

        alignment_deficit = max(samples_needed - len(all_accepted), 0)
        if alignment_deficit > 0 and "verification_deficit" not in warnings:
            warnings.append("verification_deficit")

        verification_block = _build_verification_block(
            requested=samples_needed,
            threshold=threshold,
            max_retries=AlignVerifier.MAX_RETRIES,
            attempts_used=attempts_used,
            accepted_samples=len(all_accepted),
            alignment_deficit=alignment_deficit,
            termination_reason=termination_reason,
            accepted_scores=accepted_scores,
            rejected_scores=rejected_scores,
            accepted_per_attempt=accepted_per_attempt,
            attempt_trace=attempt_trace,
            total_generated_across_retries=total_generated,
        )
        return all_accepted, verification_block, warnings


# ======================================================================
# Feature translation
# ======================================================================

def _translate_features(
    fm: FM,
    features: Dict[str, List[str]],
    or_group_mode: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Translate user-friendly ``{name: [values]}`` to internal ``fm_configuration``.

    Resolution strategy for each key in *features*:
      1. Look up the node in the FM index by **name** (not path).
      2. Determine placement based on the node's parent group type:
         - Parent is an ``alt`` or ``or`` group → ``selected_options[parent.id]``
         - Node has ``is_string_feature=True`` → ``string_values[node.id]``
         - Node is an ``and`` group → each selected child goes to ``selected_features``
         - Node is a leaf ``feature`` → each value name resolved to
           ``selected_features`` as full path
    """
    index = fm.to_dict()["index"]

    # Build a name→entry lookup (first match wins)
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in index.values():
        name = entry["name"]
        if name not in by_name:
            by_name[name] = entry

    selected_options: Dict[str, List[str]] = {}
    string_values: Dict[str, List[str]] = {}
    selected_features: List[str] = []

    for feature_name, values in features.items():
        entry = by_name.get(feature_name)
        if entry is None:
            raise ValueError(
                f"Unknown feature '{feature_name}'. "
                f"Available features: {sorted(by_name.keys())}"
            )

        group_type = entry["group_type"]

        if group_type in ("alt", "or"):
            # Node IS an alt/or group — values are option names under it
            selected_options[entry["id"]] = list(values)
        elif entry.get("is_string_feature"):
            # String-typed feature — free-text values
            string_values[entry["id"]] = list(values)
        elif group_type == "and":
            # AND group — each value is the name of a child to select
            for v in values:
                child_entry = by_name.get(v)
                if child_entry and child_entry.get("parent_id") == entry["id"]:
                    selected_features.append(child_entry["id"])
                else:
                    candidate = f"{entry['id']}.{v}"
                    if candidate in index:
                        selected_features.append(candidate)
                    else:
                        raise ValueError(
                            f"'{v}' is not a child of AND-group '{feature_name}'."
                        )
        else:
            # Leaf feature — select the feature itself
            selected_features.append(entry["id"])

    return {
        "selected_options": selected_options,
        "string_values": string_values,
        "selected_features": selected_features,
        "or_group_mode": dict(or_group_mode) if or_group_mode else {},
    }


# ======================================================================
# Report building (extracted from generation_service)
# ======================================================================

def _build_prompts_block(
    raw_samples: List[Dict[str, Any]],
    features: Dict[str, Any],
) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for sample in raw_samples:
        config = sample.get("config", {})
        prompt = config.get("prompt", "")
        if prompt not in seen:
            optimized = "optimized_prompt" in config
            entry: Dict[str, Any] = {
                "prompt": prompt,
                "features": _extract_constraint_features(config),
                "samples_produced": 0,
                "optimized": optimized,
            }
            if optimized:
                pace_score = config.get("pace_score")
                if pace_score is not None:
                    entry["pace_score"] = round(float(pace_score), 4)
                entry["pace"] = {
                    "iterations": int(features.get("pace_iterations", 0)),
                    "actors": int(features.get("pace_actors", 0)),
                    "candidates": int(features.get("pace_candidates", 0)),
                    "alpha": float(features.get("pace_alpha", 0.5)),
                }
            seen[prompt] = entry
        seen[prompt]["samples_produced"] += 1
    return list(seen.values())


def _extract_constraint_features(config: Dict[str, Any]) -> Dict[str, Any]:
    constraints = config.get("__fm_constraints__", [])
    result: Dict[str, Any] = {}
    for c in constraints:
        if not isinstance(c, dict):
            continue
        label = c.get("label", "")
        value = c.get("value")
        if label and value not in (None, "", []):
            result[label] = value
    return result


def _build_report(
    run_id: str,
    features: Dict[str, Any],
    raw_samples: List[Dict[str, Any]],
    requested: int,
    produced: int,
    warnings: List[str],
    verification_block: Any = False,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "alignment_verification": verification_block,
        "setup": {
            "model": features.get("llm", "unknown"),
            "temperature": float(features.get("temperature", 0)),
            "top_p": float(features.get("top_p", 0)),
            "prompt_approach": features.get("prompt_approach", "default"),
            "requested_samples": requested,
            "produced_samples": produced,
        },
        "prompts": _build_prompts_block(raw_samples, features),
    }
    if warnings:
        report["warnings"] = list(warnings)
    return report


def _build_verification_block(
    requested: int,
    *,
    threshold: float,
    max_retries: int,
    attempts_used: int,
    accepted_samples: int,
    alignment_deficit: int,
    termination_reason: str,
    accepted_scores: List[float],
    rejected_scores: List[float],
    accepted_per_attempt: List[int],
    attempt_trace: List[Dict[str, Any]],
    total_generated_across_retries: int,
) -> Dict[str, Any]:
    return {
        "alignment_threshold": threshold,
        "max_retries": max_retries,
        "attempts_used": attempts_used,
        "requested_samples": requested,
        "accepted_samples": accepted_samples,
        "alignment_deficit": alignment_deficit,
        "termination_reason": termination_reason,
        "scores": {
            "accepted": _score_stats(accepted_scores),
            "rejected": _score_stats(rejected_scores),
        },
        "total_generated_across_retries": total_generated_across_retries,
        "accepted_per_attempt": accepted_per_attempt,
        "attempt_trace": attempt_trace,
    }


def _score_stats(scores: List[float]) -> Dict[str, Any]:
    if not scores:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(scores),
        "min": round(min(scores), 4),
        "mean": round(sum(scores) / len(scores), 4),
        "max": round(max(scores), 4),
    }
