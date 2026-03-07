"""Shared constants and helpers for Synthline core modules."""
from typing import Any, Dict, List, Tuple

# Fields that control operation behavior rather than describing feature model data.
# Used by Promptline, OutputHandler, and Logger to separate operational concerns
# from FM-derived feature data.
OPERATING_FIELDS = frozenset({
    "llm",
    "temperature",
    "top_p",
    "samples_per_prompt",
    "total_samples",
    "prompt_approach",
    "pace_iterations",
    "pace_actors",
    "pace_candidates",
    "pace_alpha",
    "connection_id",
    "operation_id",
    "optimized_prompt",
    "optimized_atomic_prompts",
    "pace_score",
    "fm_configuration",
    "align_verify",
    "align_threshold",
    "prompt",
    "__fm_constraints__",
    "classification_label",
    "classification_label_def",
})


def extract_fm_constraints(features: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Extract (label, raw_value) pairs from __fm_constraints__, skipping empty entries.

    Shared by Promptline, PACE, and AlignScorer to ensure consistent
    constraint filtering logic.
    """
    raw_constraints = features.get("__fm_constraints__", [])
    if not isinstance(raw_constraints, list):
        return []
    results: List[Tuple[str, Any]] = []
    for item in raw_constraints:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("id") or "")
        value = item.get("value")
        if not label or value in (None, "", [], False):
            continue
        results.append((label, value))
    return results
