"""Verification engine — verify-then-regenerate loop with config-aware replacement.

Extracted from :mod:`synthline.client` to keep the SDK client a thin orchestrator
and make the verification logic independently testable.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from synthline.core.align_verifier import AlignVerifier
from synthline.core.generator import Generator
from synthline.types import Dataset, ScoredSample, VerificationCallback, VerificationEvent


# ======================================================================
# Config identity
# ======================================================================

_CONFIG_KEY_EXCLUDE = frozenset({"prompt", "optimized_prompt", "pace_score"})


def config_key(config: Dict[str, Any]) -> str:
    """Stable, hashable identity for a sample's generation config.

    Excludes derived/meta keys (``prompt``, ``optimized_prompt``,
    ``pace_score``) that do not define the atomic configuration.
    """
    filtered = {k: v for k, v in config.items() if k not in _CONFIG_KEY_EXCLUDE}
    return json.dumps(filtered, sort_keys=True, ensure_ascii=True)


# ======================================================================
# Raw-sample reconstruction (for averify)
# ======================================================================


def build_prompt_lookup(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Map feature-combo keys to prompt text from metadata["prompts"].

    Values are coerced to strings so that lookups still match after a
    CSV round-trip (``Dataset.load`` reads everything as ``str``).
    """
    lookup: Dict[str, str] = {}
    for entry in metadata.get("prompts", []):
        features = entry.get("features", {})
        key = json.dumps(
            {k: str(v) for k, v in features.items()},
            sort_keys=True, ensure_ascii=True,
        )
        lookup[key] = entry.get("prompt", "")
    return lookup


def reconstruct_raw_samples(
    samples: List[Dict[str, Any]],
    prompt_lookup: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Reconstruct internal raw samples from formatted (CSV-row) samples.

    Inverse of :meth:`Dataset.format_sample` — builds the
    ``{"text": ..., "config": {...}}`` dicts that the verifier and
    generator expect.
    """
    raw: List[Dict[str, Any]] = []
    for sample in samples:
        features = {k: v for k, v in sample.items() if k not in ("Text", "Label")}
        key = json.dumps(features, sort_keys=True, ensure_ascii=True)
        prompt_text = prompt_lookup.get(key, "")
        raw.append(Dataset.unformat_sample(sample, prompt_text))
    return raw


# ======================================================================
# Verification loop
# ======================================================================


def _group_rejected_by_config(
    rejected: List[ScoredSample],
) -> Dict[str, Tuple[Dict[str, Any], int]]:
    """Group rejected ``(sample, score)`` tuples by generation config.

    Returns ``{config_key: (representative_config, rejection_count)}``.
    """
    groups: Dict[str, Tuple[Dict[str, Any], int]] = {}
    for sample, _score in rejected:
        cfg = sample.get("config", {})
        key = config_key(cfg)
        if key in groups:
            rep, count = groups[key]
            groups[key] = (rep, count + 1)
        else:
            groups[key] = (cfg, 1)
    return groups


async def run_verification_loop(
    raw_samples: List[Dict[str, Any]],
    features: Dict[str, Any],
    threshold: float,
    samples_needed: int,
    *,
    verifier: AlignVerifier,
    generator: Generator,
    api_keys: Optional[Dict[str, str]] = None,
    on_verification: VerificationCallback = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Verify samples, regenerating rejected ones up to MAX_RETRIES.

    Returns ``(accepted_samples, alignment_verification_dict, warnings)``.
    """
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
            await on_verification(VerificationEvent(
                attempt=attempt + 1,
                max_attempts=total_attempts,
                accepted=len(all_accepted),
                needed=samples_needed,
                progress=progress,
            ))

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
            await on_verification(VerificationEvent(
                attempt=attempt + 1,
                max_attempts=total_attempts,
                accepted=len(all_accepted),
                needed=samples_needed,
                progress=progress,
            ))

        config_groups = _group_rejected_by_config(rejected)

        attempt_trace.append({
            "attempt": attempt + 1,
            "pending_in": len(pending),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "accepted_added": added,
            "rejected_configs": len(config_groups),
            "duration_ms": int((time.perf_counter() - started) * 1000),
        })

        if len(all_accepted) >= samples_needed:
            termination_reason = "count_reached"
            break
        if not rejected:
            termination_reason = "no_rejected_remaining"
            break
        if attempt == AlignVerifier.MAX_RETRIES:
            break

        # Config-aware regeneration: replace rejected samples from
        # their *originating* config.
        deficit = samples_needed - len(all_accepted)
        samples_per_prompt = int(features.get("samples_per_prompt", 1))

        config_requests: List[Tuple[Dict[str, Any], int]] = []
        remaining = deficit
        for _key, (cfg, rejected_count) in config_groups.items():
            if remaining <= 0:
                break
            count = min(rejected_count, remaining)
            config_requests.append((cfg, count))
            remaining -= count

        if not config_requests:
            termination_reason = "no_configs_to_regenerate"
            break

        regen_result = await generator.generate_for_configs(
            config_requests=config_requests,
            samples_per_prompt=samples_per_prompt,
            api_keys=api_keys,
        )
        pending = regen_result.samples
        total_generated += len(pending)

        if regen_result.fewer_samples_received and "fewer_samples_received" not in warnings:
            warnings.append("fewer_samples_received")
        if regen_result.parsing_degraded and "parsing_degraded" not in warnings:
            warnings.append("parsing_degraded")
        if not pending:
            termination_reason = "generation_returned_empty"
            break

    alignment_deficit = max(samples_needed - len(all_accepted), 0)
    if alignment_deficit > 0 and "verification_deficit" not in warnings:
        warnings.append("verification_deficit")

    alignment_verification = _build_alignment_verification(
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
    return all_accepted, alignment_verification, warnings


# ======================================================================
# Metadata helpers
# ======================================================================


def _build_alignment_verification(
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
