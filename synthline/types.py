"""Public data types for the Synthline SDK."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd

from synthline.core.constants import OPERATING_FIELDS


# ---------------------------------------------------------------------------
# Callback event types (public SDK surface)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationEvent:
    """Payload passed to a :data:`VerificationCallback` on each verification step."""

    attempt: int
    max_attempts: int
    accepted: int
    needed: int
    progress: float


@dataclass(frozen=True)
class PromptUpdateEvent:
    """Payload passed to a :data:`PromptUpdateCallback` after each PACE iteration."""

    prompt: str
    score: float
    iteration: int
    total_iterations: int
    config_index: int
    total_configs: int


# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------

ProgressCallback = Optional[Callable[[float, str], Awaitable[None]]]
"""``async (progress_pct, message) -> None``"""

VerificationCallback = Optional[Callable[[VerificationEvent], Awaitable[None]]]
"""``async (event: VerificationEvent) -> None``"""

PromptUpdateCallback = Optional[Callable[[PromptUpdateEvent], Awaitable[None]]]
"""``async (event: PromptUpdateEvent) -> None``"""

StepCallback = Optional[Callable[[], Awaitable[None]]]
"""``async () -> None`` — lightweight signal that one unit of work completed."""


# ---------------------------------------------------------------------------
# Internal result types
# ---------------------------------------------------------------------------

ScoredSample = Tuple[Dict[str, Any], float]
"""A verified sample paired with its alignment score: ``(sample_dict, score)``."""


@dataclass(frozen=True)
class OptimizedPrompt:
    """Result of PACE optimization for a single atomic configuration."""

    prompt: str
    score: float
    config: Dict[str, Any]


@dataclass
class GenerationResult:
    """Raw samples produced by :class:`Generator`."""

    samples: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# PromptEntry — one atomic prompt inside a PromptSet
# ---------------------------------------------------------------------------

@dataclass
class PromptEntry:
    """A single prompt bound to its atomic FM configuration."""

    prompt: str
    config: Dict[str, Any]
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"prompt": self.prompt, "config": self.config}
        if self.score is not None:
            d["score"] = self.score
        return d


# ---------------------------------------------------------------------------
# PromptSet — flows between build_prompts → optimize → generate
# ---------------------------------------------------------------------------

@dataclass
class PromptSet:
    """Ordered collection of atomic prompts produced by :meth:`Synthline.build_prompts`.

    Passes unchanged through :meth:`Synthline.optimize` (which replaces prompt
    text and attaches PACE scores) and into :meth:`Synthline.generate`.
    """

    entries: List[PromptEntry]
    label: str = ""
    label_definition: str = ""
    samples_per_prompt: int = 1
    optimized: bool = False
    base_features: Dict[str, Any] = field(default_factory=dict, repr=False)

    # -- sequence protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[PromptEntry]:
        return iter(self.entries)

    def __getitem__(self, index: int) -> PromptEntry:
        return self.entries[index]

    # -- conversion ---------------------------------------------------------

    def to_features_dict(
        self,
        total_samples: int,
        llm: str,
        temperature: float,
        top_p: float,
        *,
        reasoning: Optional[Dict[str, Any]] = None,
        api_keys: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Build the internal features dict that :class:`Generator` expects.

        If the prompts have been optimized, populates
        ``optimized_atomic_prompts``.  Otherwise populates
        ``fm_configuration`` so the generator builds prompts from scratch.
        """
        base: Dict[str, Any] = {
            **self.base_features,
            "llm": llm,
            "temperature": temperature,
            "top_p": top_p,
            "total_samples": total_samples,
            "samples_per_prompt": self.samples_per_prompt,
        }
        if reasoning:
            base["reasoning"] = reasoning

        if self.optimized:
            base["optimized_atomic_prompts"] = [
                {
                    "optimized_prompt": e.prompt,
                    "config": e.config,
                    **({"pace_score": e.score} if e.score is not None else {}),
                }
                for e in self.entries
            ]

        return base

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize to a JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": self.label,
            "label_definition": self.label_definition,
            "samples_per_prompt": self.samples_per_prompt,
            "optimized": self.optimized,
            "entries": [e.to_dict() for e in self.entries],
            "base_features": self.base_features,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "PromptSet":
        """Deserialize from a JSON file previously written by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = [
            PromptEntry(
                prompt=e["prompt"],
                config=e["config"],
                score=e.get("score"),
            )
            for e in data["entries"]
        ]
        return cls(
            entries=entries,
            label=data.get("label", ""),
            label_definition=data.get("label_definition", ""),
            samples_per_prompt=data.get("samples_per_prompt", 1),
            optimized=data.get("optimized", False),
            base_features=data.get("base_features", data.get("_features", {})),
        )


# ---------------------------------------------------------------------------
# Dataset — returned by Synthline.generate()
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """Generated samples with run metadata."""

    samples: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- sequence protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.samples)

    # -- formatting ---------------------------------------------------------

    @staticmethod
    def format_sample(sample_text: str, atomic_config: Dict[str, Any]) -> Dict[str, Any]:
        """Format raw sample text and atomic config into a structured dictionary."""
        sample: Dict[str, Any] = {"Text": sample_text}

        label = atomic_config.get("classification_label", "")
        if label:
            sample["Label"] = label

        for k, v in atomic_config.items():
            if k not in OPERATING_FIELDS:
                # Leaf name from dotted FM paths, e.g. "Root.X.Y" → "Y"
                col_name = k.rsplit(".", 1)[-1] if "." in k else k
                sample[col_name] = v

        return sample

    @staticmethod
    def unformat_sample(
        formatted: Dict[str, Any],
        prompt_text: str = "",
    ) -> Dict[str, Any]:
        """Inverse of :meth:`format_sample` — reconstruct internal raw sample.

        Builds the ``{"text": ..., "config": {...}}`` dict that the verifier
        and generator expect from a formatted (CSV-row-like) sample.
        """
        text = formatted.get("Text", "")
        label = formatted.get("Label", "")
        features = {k: v for k, v in formatted.items() if k not in ("Text", "Label")}
        constraints = [{"label": k, "value": v} for k, v in features.items()]
        config: Dict[str, Any] = {
            **features,
            "classification_label": label,
            "__fm_constraints__": constraints,
        }
        if prompt_text:
            config["prompt"] = prompt_text
        return {"text": text, "config": config}

    # -- export -------------------------------------------------------------

    def to_csv(self) -> str:
        """Return samples as a CSV string."""
        if not self.samples:
            return ""
        return pd.DataFrame(self.samples).to_csv(index=False, encoding="utf-8")

    def to_dataframe(self) -> pd.DataFrame:
        """Return samples as a pandas DataFrame."""
        return pd.DataFrame(self.samples)

    def save(self, path: str) -> None:
        """Write ``data.csv`` and ``metadata.json`` to *path*."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        csv_text = self.to_csv()
        if csv_text:
            (out / "data.csv").write_text(csv_text, encoding="utf-8")

        if self.metadata:
            (out / "metadata.json").write_text(
                json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    @classmethod
    def load(cls, path: str) -> "Dataset":
        """Load a Dataset from a directory written by :meth:`save`."""
        src = Path(path)

        metadata: Dict[str, Any] = {}
        meta_path = src / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        samples: List[Dict[str, Any]] = []
        csv_path = src / "data.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, encoding="utf-8", dtype=str, keep_default_na=False)
            samples = df.to_dict("records")

        return cls(samples=samples, metadata=metadata)
