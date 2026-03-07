"""
Prompt manager for Synthline.
Builds prompts for LLM calls and manages PACE optimization.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from synthline.core.constants import OPERATING_FIELDS, extract_fm_constraints
from synthline.core.fm_resolver import FMResolver
from synthline.core.fm_parser import FM
from synthline.core.align_scorer import AlignScorer
from synthline.core.llm import LLMClient
from synthline.core.pace import PACE
from synthline.utils.logger import Logger


class Promptline:
    """Builds parameterized and optionally optimized prompts for data generation."""

    _SINGLE_SUFFIX_TEMPLATE = "\n\nReturn only the raw {artefact} text. No additional text or formatting."
    _MULTI_SUFFIX_TEMPLATE = """\n\nFormat your completion as a JSON array of strings, e.g.:
[
  "First {artefact} text goes here.",
  "Second {artefact} text goes here."
]

Each string must contain only the raw {artefact} text.
Include only the JSON array. No additional text."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger: Logger,
        fm: FM,
        glossary: Optional[Dict[str, str]] = None,
        align_scorer: Optional[AlignScorer] = None,
    ):
        """Initialize the promptline manager."""
        self._llm = llm_client
        self._logger = logger
        self._fm = fm
        self._expander = FMResolver(fm=fm)
        self._glossary = glossary or {}
        self._glossary_lookup = {str(key).lower(): str(value) for key, value in self._glossary.items()}
        self._align_scorer = align_scorer
        self._pace = PACE(llm_client=llm_client, logger=logger)

    def get_atomic_configurations(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        fm_configuration = features.get("fm_configuration")
        if not isinstance(fm_configuration, dict):
            raise ValueError("fm_configuration is required and must be a dict")

        base = {
            key: value
            for key, value in features.items()
            if key not in OPERATING_FIELDS and not str(key).startswith("__")
        }
        expanded = self._expander.resolve(fm_configuration)
        configs = []
        for variant in expanded:
            config = dict(base)
            config.update(variant)
            configs.append(self._clean_config(config))
        return configs or [self._clean_config(dict(base))]

    def build(self, features: Dict[str, Any], *, samples_per_prompt: Optional[int] = None) -> str:
        """Build a generic prompt based on FM-derived constraints."""
        if samples_per_prompt is None:
            samples_per_prompt = int(features.get("samples_per_prompt", 1) or 1)
        is_multi = samples_per_prompt > 1

        artefact_singular = self._fm.artefact_type.strip() or "artefact"
        artefact_plural = self._pluralize(artefact_singular)

        if is_multi:
            lines = [
                f"Generate {samples_per_prompt} diverse {artefact_plural.lower()} satisfying the following constraints:"
            ]
        else:
            lines = [f"Generate one {artefact_singular.lower()} satisfying the following constraints:"]

        # Prepend classification context if present
        cls_label = str(features.get("classification_label", "")).strip()
        cls_def = str(features.get("classification_label_def", "")).strip()

        constraints = self._extract_constraints(features)

        all_lines: List[Dict[str, Any]] = []
        if cls_label:
            cls_text = cls_label
            if cls_def:
                cls_text += f" — {cls_def}"
            all_lines.append({"label": "ClassificationLabel", "value": cls_text, "raw_values": [cls_label]})

        all_lines.extend(constraints)

        if all_lines:
            for idx, constraint in enumerate(all_lines, start=1):
                label = constraint["label"]
                value = constraint["value"]
                definitions = self._lookup_definitions(constraint["raw_values"])
                if definitions:
                    defs_text = "; ".join(definitions)
                    lines.append(f"{idx}. {label}: {value} ({defs_text}).")
                else:
                    lines.append(f"{idx}. {label}: {value}.")

        artefact_label = artefact_singular.lower()
        suffix_template = self._MULTI_SUFFIX_TEMPLATE if is_multi else self._SINGLE_SUFFIX_TEMPLATE
        suffix = suffix_template.format(artefact=artefact_label)
        return "\n".join(lines) + suffix

    def _extract_constraints(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        formatted = []
        for label, raw_value in extract_fm_constraints(features):
            value = self._format_value(raw_value)
            if value:
                raw_values = self._raw_values(raw_value)
                formatted.append({"label": label, "value": value, "raw_values": raw_values})
        return formatted

    def _format_value(self, value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(str(v) for v in value if str(v).strip())
        if isinstance(value, bool):
            return "enabled" if value else ""
        return str(value).strip()

    def _raw_values(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if value in (None, ""):
            return []
        return [str(value).strip()]

    def _lookup_definitions(self, raw_values: List[str]) -> List[str]:
        if not self._glossary_lookup:
            return []

        definitions: List[str] = []
        seen = set()
        for raw in raw_values:
            definition = self._glossary_lookup.get(raw.lower())
            if definition and definition not in seen:
                definitions.append(definition)
                seen.add(definition)
        return definitions

    @staticmethod
    def _pluralize(noun: str) -> str:
        clean = noun.strip()
        if not clean:
            return "artefacts"
        lower = clean.lower()
        if lower.endswith(("s", "x", "z")):
            return f"{clean}es"
        if lower.endswith(("sh", "ch")):
            return f"{clean}es"
        if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
            return f"{clean[:-1]}ies"
        return f"{clean}s"

    def _clean_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in config.items()
            if value not in (None, "", [])
        }

    def get_atomic_prompts(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        atomic_configs = self.get_atomic_configurations(features)
        spp = int(features.get("samples_per_prompt", 1) or 1)
        atomic_prompts = []
        for config in atomic_configs:
            prompt = self.build(config, samples_per_prompt=spp)
            atomic_prompts.append({"config": config, "prompt": prompt})
        return atomic_prompts

    async def optimize_batch(
        self,
        atomic_configs,
        features,
        progress_callback=None,
        prompt_update_callback=None,
        api_keys=None,
    ):
        pace_alpha = float(features.get("pace_alpha", 0.5))
        self._pace.set_align_scorer(
            self._align_scorer if pace_alpha > 0.0 else None
        )
        return await self._pace.optimize_batch(
            atomic_configs=atomic_configs,
            features=features,
            progress_callback=progress_callback,
            n_iterations=int(features.get("pace_iterations")),
            n_actors=int(features.get("pace_actors")),
            n_candidates=int(features.get("pace_candidates")),
            prompt_update_callback=prompt_update_callback,
            api_keys=api_keys,
        )


