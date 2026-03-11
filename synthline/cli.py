"""Synthline CLI — generate synthetic data from the command line."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _progress_bar(progress: float, message: str, *, width: int = 30) -> str:
    """Format a single-line progress bar: [████████░░░░░░] 54% message."""
    pct = max(0.0, min(100.0, progress))
    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"\r  [{bar}] {pct:3.0f}% {message}"


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="synthline",
        description="Synthline — Feature Model–Guided Synthetic Data Generator",
    )
    sub = parser.add_subparsers(dest="command")

    # -- build-prompts ------------------------------------------------------
    bp = sub.add_parser("build-prompts", help="Build atomic prompts (no LLM call)")
    _add_common_args(bp)

    # -- optimize -----------------------------------------------------------
    opt = sub.add_parser("optimize", help="Optimize prompts via PACE")
    _add_common_args(opt)
    opt.add_argument("--llm", required=True, help="LLM model identifier")
    opt.add_argument("--temperature", type=float, default=1.0)
    opt.add_argument("--top-p", type=float, default=1.0)
    opt.add_argument("--alpha", type=float, default=0.5, help="PACE alpha")
    opt.add_argument("--iterations", type=int, default=1, help="PACE iterations")
    opt.add_argument("--actors", type=int, default=4, help="PACE actors")
    opt.add_argument("--candidates", type=int, default=2, help="PACE candidates")
    opt.add_argument("--output", "-o", required=True, help="Output directory")

    # -- generate -----------------------------------------------------------
    gen = sub.add_parser("generate", help="Generate synthetic data")
    _add_common_args(gen)
    gen.add_argument("--prompts", help="Path to saved PromptSet JSON (skip build)")
    gen.add_argument("--llm", help="LLM model identifier")
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--samples", type=int, help="Number of samples (required unless --verify-input)")
    gen.add_argument("--verify", action="store_true", help="Enable alignment verification")
    gen.add_argument("--verify-threshold", type=float, default=0.5)
    gen.add_argument("--verify-input", help="Path to existing dataset to verify (skip generation)")
    gen.add_argument("--output", "-o", required=True, help="Output directory")

    # -- validate -----------------------------------------------------------
    sub.add_parser("validate", help="Validate a feature model").add_argument(
        "--fm", required=True, help="Path to fm.xml"
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "validate":
        _cmd_validate(args)
    elif args.command == "build-prompts":
        _cmd_build_prompts(args)
    elif args.command == "optimize":
        _cmd_optimize(args)
    elif args.command == "generate":
        _cmd_generate(args)


# ---------------------------------------------------------------------------
# Common arguments
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fm", help="Path to fm.xml")
    parser.add_argument("--glossary", help="Path to glossary.yaml")
    parser.add_argument("--label", help="Classification label")
    parser.add_argument("--label-def", default="", help="Classification label definition")
    parser.add_argument("--features", help="Path to features YAML")
    parser.add_argument("--samples-per-prompt", type=int, default=50)
    parser.add_argument("--config", help="Path to a YAML config file bundling all params")
    parser.add_argument("--debug", action="store_true")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_validate(args: argparse.Namespace) -> None:
    from synthline.core.fm_parser import FM

    try:
        fm = FM(args.fm)
        index = fm.to_dict()["index"]
        print(f"Valid feature model: {fm.artefact_type}")
        print(f"  Nodes: {len(index)}")
        print(f"  Constraints: {len(fm.constraints)}")
    except Exception as exc:
        print(f"Invalid feature model: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_build_prompts(args: argparse.Namespace) -> None:
    cfg = _resolve_config(args)
    sl = _create_client(cfg)
    prompts = sl.build_prompts(
        label=cfg["label"],
        label_definition=cfg.get("label_definition", ""),
        samples_per_prompt=cfg.get("samples_per_prompt", 50),
        features=cfg["features"],
    )

    print(f"Built {len(prompts)} atomic prompt(s):")
    for i, entry in enumerate(prompts):
        print(f"\n--- Prompt {i + 1} ---")
        print(entry.prompt[:200] + ("..." if len(entry.prompt) > 200 else ""))

    output = cfg.get("output")
    if output:
        path = str(Path(output) / "prompts.json")
        prompts.save(path)
        print(f"\nSaved to {path}")


def _cmd_optimize(args: argparse.Namespace) -> None:
    cfg = _resolve_config(args)
    sl = _create_client(cfg)

    prompts = sl.build_prompts(
        label=cfg["label"],
        label_definition=cfg.get("label_definition", ""),
        samples_per_prompt=cfg.get("samples_per_prompt", 50),
        features=cfg["features"],
    )
    print(f"Built {len(prompts)} atomic prompt(s). Optimizing...")

    last_pct = [-1]

    async def _on_progress(progress: float, message: str) -> None:
        pct = int(progress)
        if pct == last_pct[0]:
            return
        last_pct[0] = pct
        print(_progress_bar(progress, message), end="", flush=True)

    optimized = asyncio.run(
        sl.aoptimize(
            prompts,
            alpha=cfg.get("alpha", 0.5),
            iterations=cfg.get("iterations", 1),
            actors=cfg.get("actors", 4),
            candidates=cfg.get("candidates", 2),
            on_progress=_on_progress,
        )
    )
    print()

    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    optimized.save(str(out / "prompts.json"))
    print(f"Saved {len(optimized)} optimized prompt(s) to {out / 'prompts.json'}")
    for i, entry in enumerate(optimized):
        score_str = f" (score: {entry.score:.4f})" if entry.score is not None else ""
        print(f"  Prompt {i + 1}{score_str}")


def _cmd_generate(args: argparse.Namespace) -> None:
    cfg = _resolve_config(args)

    last_pct = [-1]

    async def _on_progress(progress: float, message: str) -> None:
        pct = int(progress)
        if pct == last_pct[0]:
            return
        last_pct[0] = pct
        print(_progress_bar(progress, message), end="", flush=True)

    # -- verify-input: shared-base workflow (skip generation) ---------------
    if cfg.get("verify_input"):
        from synthline.types import Dataset

        dataset = Dataset.load(cfg["verify_input"])
        print(f"Loaded {len(dataset)} samples from {cfg['verify_input']}")

        sl = _create_client(cfg)
        output = asyncio.run(
            sl.averify(
                dataset,
                threshold=cfg.get("verify_threshold", 0.5),
                on_progress=_on_progress,
            )
        )
        print()

        out = Path(cfg["output"])
        output.save(str(out))
        print(f"Verified {len(output)} samples → {out}/")
        return

    # -- standard generation path ------------------------------------------
    if not cfg.get("samples"):
        print("Error: --samples is required for generation", file=sys.stderr)
        sys.exit(1)

    if cfg.get("prompts"):
        from synthline.types import PromptSet

        prompts = PromptSet.load(cfg["prompts"])
        print(f"Loaded {len(prompts)} prompt(s) from {cfg['prompts']}")
    else:
        sl = _create_client(cfg)
        prompts = sl.build_prompts(
            label=cfg["label"],
            label_definition=cfg.get("label_definition", ""),
            samples_per_prompt=cfg.get("samples_per_prompt", 50),
            features=cfg["features"],
        )
        print(f"Built {len(prompts)} atomic prompt(s)")

    sl = _create_client(cfg)

    output = asyncio.run(
        sl.agenerate(
            prompts,
            samples=cfg["samples"],
            verify=cfg.get("verify", False),
            verify_threshold=cfg.get("verify_threshold", 0.5),
            on_progress=_on_progress,
        )
    )
    print()

    out = Path(cfg["output"])
    output.save(str(out))
    print(f"Generated {len(output)} samples → {out}/")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI args with an optional --config YAML file."""
    cfg: Dict[str, Any] = {}

    if getattr(args, "config", None):
        data = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cfg.update(data)

    # CLI args override YAML
    for key in (
        "fm", "glossary", "label", "label_def", "features",
        "samples_per_prompt", "llm", "temperature", "top_p",
        "samples", "verify", "verify_threshold", "verify_input", "output",
        "alpha", "iterations", "actors", "candidates",
        "prompts", "debug",
    ):
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            cfg[key] = val

    # Rename label_def → label_definition
    if "label_def" in cfg:
        cfg.setdefault("label_definition", cfg.pop("label_def"))

    # Load features YAML if it's a file path
    if isinstance(cfg.get("features"), str):
        features_path = Path(cfg["features"])
        if features_path.exists():
            data = yaml.safe_load(features_path.read_text(encoding="utf-8"))
            cfg["features"] = data if isinstance(data, dict) else {}

    return cfg


def _create_client(cfg: Dict[str, Any]) -> "Synthline":
    from synthline.client import Synthline

    if not cfg.get("fm"):
        print("Error: --fm is required", file=sys.stderr)
        sys.exit(1)

    return Synthline(
        fm=cfg["fm"],
        llm=cfg.get("llm", ""),
        glossary=cfg.get("glossary"),
        temperature=float(cfg.get("temperature", 1.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        debug=cfg.get("debug", False),
    )


if __name__ == "__main__":
    main()
