"""
Alignment verifier for Synthline.
Post-generation quality gate using NLI-based text-attribute alignment scoring.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from synthline.core.align_scorer import AlignScorer
from synthline.utils.logger import Logger

ScoredSample = Tuple[Dict[str, Any], float]


class AlignVerifier:
    """Score generated samples and split them into accepted/rejected based on a threshold."""

    MAX_RETRIES = 3

    def __init__(self, align_scorer: AlignScorer, logger: Logger) -> None:
        self._align_scorer = align_scorer
        self._logger = logger

    def verify(
        self,
        samples: List[Dict[str, Any]],
        threshold: float,
    ) -> Tuple[List[ScoredSample], List[ScoredSample]]:
        """Score and split samples into (accepted, rejected).

        Each sample dict must have 'text' and 'config' keys.
        Config must contain '__fm_constraints__' for NLI scoring.

        Accepted list is sorted by score descending (best first).
        """
        if not samples:
            return [], []

        scored = self._score_all(samples)

        accepted: List[ScoredSample] = []
        rejected: List[ScoredSample] = []

        for scored_sample in scored:
            if scored_sample[1] >= threshold:
                accepted.append(scored_sample)
            else:
                rejected.append(scored_sample)

        accepted.sort(key=lambda s: s[1], reverse=True)

        return accepted, rejected

    def _score_all(self, samples: List[Dict[str, Any]]) -> List[ScoredSample]:
        """Score all samples, grouping by constraint set for efficient batched NLI."""
        groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}

        for idx, sample in enumerate(samples):
            config = sample.get("config", {})
            constraints = config.get("__fm_constraints__", [])
            group_key = json.dumps(constraints, sort_keys=True, ensure_ascii=True)
            groups.setdefault(group_key, []).append((idx, sample))

        scored: List[ScoredSample] = [None] * len(samples)  # type: ignore[list-item]

        for _group_key, group_items in groups.items():
            texts = [item[1]["text"] for item in group_items]
            attributes = group_items[0][1].get("config", {})

            try:
                scores = self._align_scorer.score_samples(texts, attributes)
            except Exception as exc:
                self._logger.log_error(
                    f"AlignVerifier scoring failed: {exc}",
                    "align_verifier",
                )
                scores = [0.0] * len(texts)

            for i, (orig_idx, sample) in enumerate(group_items):
                scored[orig_idx] = (sample, scores[i])

        return scored
