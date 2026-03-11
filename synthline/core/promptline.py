"""
Prompt manager for Synthline.
Builds parameterized prompts from FM-derived feature constraints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from synthline.core.constants import LABEL_FIELDS, OPERATING_FIELDS, extract_fm_constraints
from synthline.core.fm_resolver import FMResolver
from synthline.core.fm_parser import FM


class Promptline:
    """Builds parameterized prompts for data generation."""

    def __init__(
        self,
        fm: FM,
        glossary: Optional[Dict[str, str]] = None,
    ):
        self._fm = fm
        self._resolver = FMResolver(fm=fm)
        self._glossary = glossary or {}
        self._glossary_lookup = {str(key).lower(): str(value) for key, value in self._glossary.items()}

    def get_atomic_configurations(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        fm_configuration = features.get("fm_configuration")
        if not isinstance(fm_configuration, dict):
            raise ValueError("fm_configuration is required and must be a dict")

        base = {
            key: value
            for key, value in features.items()
            if (key not in OPERATING_FIELDS or key in LABEL_FIELDS)
            and not str(key).startswith("__")
        }
        expanded = self._resolver.resolve(fm_configuration)
        configs = []
        for variant in expanded:
            config = dict(base)
            config.update(variant)
            configs.append(self._clean_config(config))
        return configs or [self._clean_config(dict(base))]

    def build(self, features: Dict[str, Any], *, samples_per_prompt: Optional[int] = None) -> str:
        """Build a generic prompt based on FM-derived constraints."""
        if samples_per_prompt is None:
            samples_per_prompt = max(1, int(features.get("samples_per_prompt") or 1))
        plural = samples_per_prompt > 1

        artefact_singular = self._fm.artefact_type.strip() or "artefact"
        artefact_plural = self._pluralize(artefact_singular)

        if plural:
            lines = [
                f"Generate {samples_per_prompt} diverse {artefact_plural.lower()} satisfying the following constraints:"
            ]
        else:
            lines = [f"Generate one {artefact_singular.lower()} satisfying the following constraints:"]

        label = str(features.get("classification_label", "")).strip()
        label_def = str(features.get("classification_label_def", "")).strip()

        constraints = self._extract_constraints(features)

        all_lines: List[Dict[str, Any]] = []
        if label:
            label_text = label
            if label_def:
                label_text += f" — {label_def}"
            all_lines.append({"label": "ClassificationLabel", "value": label_text, "raw_values": [label]})

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

        return "\n".join(lines)

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
            return [s for v in value if (s := str(v).strip())]
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
        samples_per_prompt = max(1, int(features.get("samples_per_prompt") or 1))
        atomic_prompts = []
        for config in atomic_configs:
            prompt = self.build(config, samples_per_prompt=samples_per_prompt)
            atomic_prompts.append({"config": config, "prompt": prompt})
        return atomic_prompts


