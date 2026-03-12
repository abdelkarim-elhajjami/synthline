"""
Shared NLI-based text-attribute alignment scorer for Synthline.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from synthline.core.constants import extract_fm_constraints
from synthline.utils.logger import Logger


class AlignScorer:
    """Compute average entailment probability between samples and verbalized attributes."""

    DEFAULT_MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    ENTAILMENT_CLASS_INDEX = 0
    BATCH_SIZE = 64

    def __init__(self, logger: Logger, model_name: str) -> None:
        self._logger = logger
        self._model_name = (model_name or "").strip()
        if not self._model_name:
            raise ValueError("AlignScorer requires a non-empty NLI model name.")

        self._model = None
        self._tokenizer = None
        self._entailment_index = None
        self._model_load_attempted = False
        self._hypothesis_cache: Dict[str, str] = {}

    def score_samples(self, samples: List[str], attributes: Dict[str, Any]) -> List[float]:
        """Return per-sample entailment probabilities."""
        non_empty_samples = [s for sample in samples if (s := str(sample).strip())]
        if not non_empty_samples:
            raise ValueError("AlignScorer received an empty sample batch.")

        attribute_hypothesis = self._get_cached_hypothesis(attributes)
        pairs: List[Tuple[str, str]] = [
            (sample, attribute_hypothesis) for sample in non_empty_samples
        ]

        model, tokenizer, entailment_index = self._get_runtime_components()

        import torch

        all_probs: List[float] = []
        for i in range(0, len(pairs), self.BATCH_SIZE):
            batch_pairs = pairs[i : i + self.BATCH_SIZE]
            encoded = tokenizer(
                batch_pairs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                probs = torch.softmax(model(**encoded).logits, dim=1)[:, entailment_index]
                all_probs.extend(probs.cpu().tolist())

        return all_probs

    def score_batch(self, samples: List[str], attributes: Dict[str, Any]) -> float:
        """
        Return average NLI entailment probability over the batch.

        Each generated sample is used as premise. The verbalized attribute statement
        is used as hypothesis.
        """
        scores = self.score_samples(samples, attributes)
        return sum(scores) / len(scores)

    def _build_attribute_hypothesis(self, constraint_statements: List[str]) -> str:
        constraints_text = "; ".join(constraint_statements)
        return f"The text satisfies all of the following constraints: {constraints_text}."

    def _get_cached_hypothesis(self, attributes: Dict[str, Any]) -> str:
        constraint_statements = self._validate_and_extract_constraint_fragments(attributes)
        cache_key = self._build_hypothesis_cache_key(constraint_statements)
        cached = self._hypothesis_cache.get(cache_key)
        if cached is not None:
            return cached

        attribute_hypothesis = self._build_attribute_hypothesis(constraint_statements)
        self._hypothesis_cache[cache_key] = attribute_hypothesis
        return attribute_hypothesis

    def _build_hypothesis_cache_key(self, constraint_statements: List[str]) -> str:
        return json.dumps(sorted(constraint_statements), ensure_ascii=True)

    def _validate_and_extract_constraint_fragments(
        self,
        attributes: Dict[str, Any],
    ) -> List[str]:
        if not isinstance(attributes, dict):
            raise ValueError("AlignScorer attributes must be a dictionary.")

        constraint_statements = self._extract_constraint_fragments(attributes)
        if not constraint_statements:
            raise ValueError(
                "AlignScorer requires non-empty constraints to verbalize alignment attributes."
            )
        return constraint_statements

    def _extract_constraint_fragments(self, attributes: Dict[str, Any]) -> List[str]:
        statements: List[str] = []
        for label, value in extract_fm_constraints(attributes):
            if isinstance(value, list):
                values = [s for v in value if (s := str(v).strip())]
                if not values:
                    continue
                value_text = ", ".join(values)
            else:
                value_text = str(value).strip()
                if not value_text:
                    continue
            statements.append(f"{label} is {value_text}")
        return statements

    def _get_runtime_components(self) -> Tuple[Any, Any, int]:
        if self._model_load_attempted:
            if self._model is None or self._tokenizer is None or self._entailment_index is None:
                raise RuntimeError("AlignScorer runtime is not available after initialization failure.")
            return self._model, self._tokenizer, self._entailment_index

        self._model_load_attempted = True
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            model = model.eval()

            self._model = model
            self._tokenizer = tokenizer
            self._entailment_index = self.ENTAILMENT_CLASS_INDEX
            return self._model, self._tokenizer, self._entailment_index
        except Exception as exc:
            self._logger.log_error(
                f"Failed to initialize AlignScorer model '{self._model_name}': {exc}",
                "align_scorer",
            )
            self._model = None
            self._tokenizer = None
            self._entailment_index = None
            raise RuntimeError(
                f"AlignScorer initialization failed for model '{self._model_name}'."
            ) from exc
