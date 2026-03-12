"""Synthline SDK client — the public entry point."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from synthline.core.fm_parser import FM
from synthline.core.verification import (
    build_prompt_lookup,
    reconstruct_raw_samples,
    run_verification_loop,
)

from synthline._runtime import Runtime, create_runtime
from synthline.types import (
    Dataset,
    ProgressCallback,
    PromptEntry,
    PromptSet,
    PromptUpdateCallback,
    VerificationCallback,
)


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
        if _raw_fm_configuration:
            raw_features = dict(features)
        else:
            fm_configuration = _translate_features(self._runtime.fm, features, or_group_mode)
            raw_features = {
                "fm_configuration": fm_configuration,
                "classification_label": label,
                "classification_label_def": label_definition,
                "samples_per_prompt": samples_per_prompt,
                "llm": self._llm,
                "temperature": self._temperature,
                "top_p": self._top_p,
            }

        atomic_prompts = self._runtime.promptline.get_atomic_prompts(raw_features)

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
            base_features=raw_features,
        )

    # ======================================================================
    # optimize
    # ======================================================================

    async def optimize(
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
        # Rebuild atomic configs with embedded prompts
        atomic_configs: List[Dict[str, Any]] = []
        for entry in prompts.entries:
            config = dict(entry.config)
            config["prompt"] = entry.prompt
            atomic_configs.append(config)

        features: Dict[str, Any] = {
            **prompts.base_features,
            "llm": self._llm,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "pace_alpha": alpha,
            "pace_iterations": iterations,
            "pace_actors": actors,
            "pace_candidates": candidates,
            "prompt_approach": "PACE",
        }

        self._runtime.pace.set_align_scorer(
            self._runtime.align_scorer if alpha > 0.0 else None
        )

        optimized_results = await self._runtime.pace.optimize_batch(
            atomic_configs=atomic_configs,
            features=features,
            progress_callback=on_progress,
            n_iterations=iterations,
            n_actors=actors,
            n_candidates=candidates,
            prompt_update_callback=on_prompt_update,
            api_keys=self._api_keys,
        )

        entries = [
            PromptEntry(
                prompt=result.prompt,
                config=result.config,
                score=result.score,
            )
            for result in optimized_results
        ]

        return PromptSet(
            entries=entries,
            label=prompts.label,
            label_definition=prompts.label_definition,
            samples_per_prompt=prompts.samples_per_prompt,
            optimized=True,
            base_features=prompts.base_features,
        )

    # ======================================================================
    # generate
    # ======================================================================

    async def generate(
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
        started = time.perf_counter()
        run_id = str(uuid4())

        features = prompts.to_features_dict(
            total_samples=samples,
            llm=self._llm,
            temperature=self._temperature,
            top_p=self._top_p,
            api_keys=self._api_keys,
        )

        # --- progress adapter (scale to 0–80% when verification follows) --------
        gen_progress_cb = on_progress
        if on_progress and verify:
            async def gen_progress_cb(progress: float, message: str) -> None:
                await on_progress(progress * 0.8, message)

        # --- generation --------------------------------------------------------
        gen_result = await self._runtime.generator.generate(
            features=features,
            progress_callback=gen_progress_cb,
            api_keys=self._api_keys,
        )
        raw_samples = gen_result.samples

        # --- verification (optional) -------------------------------------------
        warnings: List[str] = []
        alignment_verification: Any = False

        if verify and raw_samples:
            if on_progress:
                await on_progress(80.0, "Verifying alignment")

            raw_samples, alignment_verification, verification_warnings = (
                await run_verification_loop(
                    raw_samples=raw_samples,
                    features=features,
                    threshold=verify_threshold,
                    samples_needed=samples,
                    verifier=self._runtime.align_verifier,
                    generator=self._runtime.generator,
                    api_keys=self._api_keys,
                    on_verification=on_verification,
                )
            )
            warnings.extend(verification_warnings)
        else:
            deficit = max(samples - len(raw_samples), 0)
            if deficit > 0:
                warnings.append("generation_deficit")

        # --- format & assemble -------------------------------------------------
        formatted = [
            Dataset.format_sample(s["text"], s["config"])
            for s in raw_samples
        ]

        metadata: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "llm": self._llm,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "samples_requested": samples,
            "samples_produced": len(formatted),
            "samples_per_prompt": int(features.get("samples_per_prompt", 1)),
            "verify": verify,
            "verify_threshold": verify_threshold if verify else None,
            "optimized": prompts.optimized,
            "prompt_approach": features.get("prompt_approach", "default"),
            "duration_seconds": round(time.perf_counter() - started, 2),
            "alignment_verification": alignment_verification,
            "prompts": _build_prompts_summary(raw_samples, features),
        }
        if warnings:
            metadata["warnings"] = list(warnings)

        if on_progress:
            await on_progress(100.0, "Complete")

        return Dataset(samples=formatted, metadata=metadata)

    # ======================================================================
    # verify (shared-base workflow)
    # ======================================================================

    async def verify(
        self,
        dataset: Dataset,
        *,
        threshold: float = 0.5,
        on_progress: ProgressCallback = None,
        on_verification: VerificationCallback = None,
    ) -> Dataset:
        """Verify a pre-generated Dataset, regenerating rejected samples.

        Reconstructs the internal raw-sample format from
        ``dataset.samples`` + ``dataset.metadata["prompts"]`` so that no
        intermediate representation needs to be persisted.
        """
        started = time.perf_counter()
        run_id = str(uuid4())

        # Reconstruct raw samples from formatted samples + metadata prompts.
        # CSV serialisation strips operating fields (llm, temperature, top_p)
        # so we restore them from the instance for config-aware regeneration.
        prompt_lookup = build_prompt_lookup(dataset.metadata)
        raw_samples = reconstruct_raw_samples(dataset.samples, prompt_lookup)
        llm_settings = {"llm": self._llm, "temperature": self._temperature, "top_p": self._top_p}
        for sample in raw_samples:
            sample["config"].update(llm_settings)

        samples_needed = len(raw_samples)
        samples_per_prompt = dataset.metadata.get("samples_per_prompt", None)
        if samples_per_prompt is None:
            n_prompts = len(dataset.metadata.get("prompts", []))
            n_samples = len(dataset.samples)
            samples_per_prompt = max(1, n_samples // n_prompts) if n_prompts > 0 else 1
        features: Dict[str, Any] = {"samples_per_prompt": samples_per_prompt}

        if on_progress:
            await on_progress(0.0, "Verifying alignment")

        accepted, alignment_verification, verification_warnings = (
            await run_verification_loop(
                raw_samples=raw_samples,
                features=features,
                threshold=threshold,
                samples_needed=samples_needed,
                verifier=self._runtime.align_verifier,
                generator=self._runtime.generator,
                api_keys=self._api_keys,
                on_verification=on_verification,
            )
        )

        formatted = [
            Dataset.format_sample(s["text"], s["config"])
            for s in accepted
        ]

        warnings: List[str] = list(verification_warnings)

        metadata: Dict[str, Any] = {
            **dataset.metadata,
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "verify": True,
            "verify_threshold": threshold,
            "samples_produced": len(formatted),
            "alignment_verification": alignment_verification,
            "source_run_id": dataset.metadata.get("run_id"),
            "duration_seconds": round(time.perf_counter() - started, 2),
        }
        if warnings:
            metadata["warnings"] = warnings

        if on_progress:
            await on_progress(100.0, "Complete")

        return Dataset(samples=formatted, metadata=metadata)


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
# Metadata helpers
# ======================================================================

def _build_prompts_summary(
    raw_samples: List[Dict[str, Any]],
    features: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Deduplicate raw samples by prompt and collect per-prompt metadata."""
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

