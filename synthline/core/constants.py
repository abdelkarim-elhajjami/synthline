"""Shared constants and helpers for Synthline core modules."""
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Semantic field groups — each frozenset collects features dict keys that
# belong to a single concern.  OPERATING_FIELDS is the union of all groups
# and is the only set that downstream code (Promptline, Dataset, Logger)
# should use to filter out non-FM-derived keys.
# ---------------------------------------------------------------------------

LLM_FIELDS = frozenset({
    "llm",
    "temperature",
    "top_p",
})

GENERATION_FIELDS = frozenset({
    "samples_per_prompt",
    "total_samples",
})

PACE_FIELDS = frozenset({
    "prompt_approach",
    "pace_iterations",
    "pace_actors",
    "pace_candidates",
    "pace_alpha",
})

ALIGNMENT_FIELDS = frozenset({
    "align_verify",
    "align_threshold",
})

LABEL_FIELDS = frozenset({
    "classification_label",
    "classification_label_def",
})

SESSION_FIELDS = frozenset({
    "connection_id",
    "operation_id",
})

INTERNAL_FIELDS = frozenset({
    "optimized_prompt",
    "optimized_atomic_prompts",
    "pace_score",
    "fm_configuration",
    "prompt",
    "__fm_constraints__",
})

# Union of all groups — used by Promptline, Dataset, and Logger to separate
# operational concerns from FM-derived feature data.
OPERATING_FIELDS = (
    LLM_FIELDS
    | GENERATION_FIELDS
    | PACE_FIELDS
    | ALIGNMENT_FIELDS
    | LABEL_FIELDS
    | SESSION_FIELDS
    | INTERNAL_FIELDS
)


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
