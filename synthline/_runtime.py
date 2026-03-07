"""Headless wiring of Synthline core modules — no FastAPI, no WebSocket, no sessions."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from synthline.core.align_scorer import AlignScorer
from synthline.core.align_verifier import AlignVerifier
from synthline.core.fm_parser import FM
from synthline.core.generator import Generator
from synthline.core.llm import LLMClient
from synthline.core.output_handler import OutputHandler
from synthline.core.promptline import Promptline
from synthline.utils.logger import Logger


@dataclass(frozen=True)
class Runtime:
    """All wired core objects needed to run Synthline operations."""

    fm: FM
    glossary: Dict[str, str]
    llm_client: LLMClient
    promptline: Promptline
    generator: Generator
    align_scorer: AlignScorer
    align_verifier: AlignVerifier
    output_handler: OutputHandler
    logger: Logger


def create_runtime(
    fm_path: str,
    glossary_path: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None,
    *,
    debug: bool = False,
) -> Runtime:
    """Create a fully wired Runtime from file paths and optional API keys.

    API keys are resolved in order:
      1. Explicit *api_keys* dict (``{"openrouter": "sk-..."}``).
      2. Environment variables (``OPENROUTER_API_KEY``, ``OPENAI_API_KEY``,
         ``OLLAMA_BASE_URL``, ``HF_TOKEN``).
    """
    logger = Logger(debug_mode=debug)

    # --- FM ---
    fm = FM(str(fm_path))

    # --- Glossary ---
    glossary = _load_glossary(glossary_path) if glossary_path else {}

    # --- LLM client ---
    keys = api_keys or {}
    llm_client = LLMClient(
        logger=logger,
        openai_key=keys.get("openai") or os.environ.get("OPENAI_API_KEY"),
        openrouter_key=keys.get("openrouter") or os.environ.get("OPENROUTER_API_KEY"),
        ollama_base_url=keys.get("ollama_base_url") or os.environ.get("OLLAMA_BASE_URL"),
        hf_token=keys.get("hf_token") or os.environ.get("HF_TOKEN"),
    )

    # --- Scorer & Verifier ---
    align_scorer = AlignScorer(
        logger=logger,
        model_name=AlignScorer.DEFAULT_MODEL_NAME,
    )
    align_verifier = AlignVerifier(
        align_scorer=align_scorer,
        logger=logger,
    )

    # --- Promptline & Generator ---
    promptline = Promptline(
        llm_client=llm_client,
        logger=logger,
        fm=fm,
        glossary=glossary,
        align_scorer=align_scorer,
    )
    generator = Generator(
        llm=llm_client,
        promptline=promptline,
        logger=logger,
    )

    # --- Output handler ---
    output_handler = OutputHandler(logger=logger)

    return Runtime(
        fm=fm,
        glossary=glossary,
        llm_client=llm_client,
        promptline=promptline,
        generator=generator,
        align_scorer=align_scorer,
        align_verifier=align_verifier,
        output_handler=output_handler,
        logger=logger,
    )


def runtime_from_deps(deps: Any) -> Runtime:
    """Create a Runtime by reusing objects from a Dependencies instance.

    Used by Web UI service adapters to avoid re-initializing heavy objects
    (FM parser, NLI model, LLM clients) that already live in the DI container.
    """
    return Runtime(
        fm=deps.fm,
        glossary=deps.glossary,
        llm_client=deps.llm_client,
        promptline=deps.promptline,
        generator=deps.generator,
        align_scorer=deps.align_scorer,
        align_verifier=deps.align_verifier,
        output_handler=deps.output_handler,
        logger=deps.logger,
    )


def _load_glossary(glossary_path: str) -> Dict[str, str]:
    path = Path(glossary_path)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v is not None}
