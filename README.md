<p align="center">
  <img src="https://raw.githubusercontent.com/abdelkarim-elhajjami/Synthline/main/docs/header.svg" alt="Synthline" width="900"/>
</p>

<p align="center">
  <a href="https://github.com/abdelkarim-elhajjami/Synthline/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-7B00FF" alt="License"></a>
  <a href="https://pypi.org/project/synthline/"><img src="https://img.shields.io/pypi/v/synthline?color=7B00FF" alt="PyPI"></a>
  <a href="https://pypi.org/project/synthline/"><img src="https://img.shields.io/pypi/pyversions/synthline?color=7B00FF" alt="Python"></a>
</p>

Generate synthetic text classification datasets whose structure is governed by a [FeatureIDE](https://featureide.github.io/) feature model.
Domain constraints are formalized, validated, and enforced — before any text is produced.

---

## What Can You Do

- **Generate constrained synthetic data** from a feature model that defines valid attribute combinations for your domain — no real data required.
- **Optimize prompts** with PACE (Prompt Actor-Critic Editing) to maximize diversity and text-attribute alignment before generation.
- **Verify alignment** with an NLI-based quality gate that checks each instance against its conditioning attributes, with automatic retry on mismatch.
- **Use any LLM** — OpenAI, OpenRouter, Ollama (local), HuggingFace Inference API.
- **Export results** as CSV, pandas DataFrames, or artifact directories.

---

## Architecture

Synthline follows the two-phase paradigm of Software Product Line Engineering. A feature model is built once per domain; datasets are derived per generation run.

<p align="center">
  <img src="https://raw.githubusercontent.com/abdelkarim-elhajjami/Synthline/main/docs/methodology.png" alt="Synthline methodology" width="750"/>
</p>

The generation pipeline translates valid FM configurations into prompts, optionally optimizes them via PACE, generates text through an LLM, and optionally verifies alignment with an NLI scorer.

<p align="center">
  <img src="https://raw.githubusercontent.com/abdelkarim-elhajjami/Synthline/main/docs/architecture.png" alt="Synthline architecture" width="750"/>
</p>

---

## Installation

```bash
pip install synthline
```

From source:

```bash
git clone https://github.com/abdelkarim-elhajjami/Synthline.git
cd Synthline
pip install -e .
```

## Quick Start

```python
from synthline import Synthline

sl = Synthline(
    fm="path/to/fm.xml",
    llm="openrouter/meta-llama/llama-3.3-70b-instruct",
    glossary="path/to/glossary.yaml",  # optional
)

# 1. Build prompts from feature selection (no LLM call)
prompts = sl.build_prompts(
    label="Security",
    label_definition="Degree to which a product protects information and data.",
    samples_per_prompt=50,
    features={
        "RequirementType": ["Quality"],
        "Domain": ["Healthcare Information System"],
        "AbstractionLevel": ["HighLevel", "DetailedLevel"],
        "DescriptionType": ["ProseNL"],
        "Context": ["Usage", "ITSystem"],
        "Language": ["EN"],
    },
)

# 2. Generate
output = sl.generate(prompts=prompts, samples=1000)

# 3. Export
output.save("output/")       # data.csv, metadata.json
df = output.to_dataframe()   # pandas DataFrame
```

### With PACE Optimization

```python
optimized = sl.optimize(prompts, alpha=0.5, iterations=1, actors=4, candidates=2)
output = sl.generate(prompts=optimized, samples=1000)
```

### With Alignment Verification

```python
output = sl.generate(prompts=prompts, samples=1000, verify=True, verify_threshold=0.6)
```

### Async API

`optimize` and `generate` have async counterparts: `aoptimize`, `agenerate`. `build_prompts` is synchronous (no LLM call).

---

## CLI

```bash
# Validate a feature model
synthline validate --fm fm.xml

# Build and inspect prompts
synthline build-prompts --fm fm.xml --label Security --label-def "..." --features features.yaml

# Optimize prompts with PACE
synthline optimize --fm fm.xml --llm openrouter/... --label Security --features features.yaml --output optimized/

# Generate synthetic data
synthline generate --fm fm.xml --llm openrouter/... --samples 1000 --verify --output out/

# Generate from a config file
synthline generate --config run.yaml --output out/
```

---

## LLM Providers

| Provider    | Prefix            | Environment variable |
| ----------- | ----------------- | -------------------- |
| OpenAI      | `openai/...`      | `OPENAI_API_KEY`     |
| OpenRouter  | `openrouter/...`  | `OPENROUTER_API_KEY` |
| Ollama      | `ollama/...`      | `OLLAMA_BASE_URL` (local) |
| HuggingFace | `huggingface/...` | `HF_TOKEN`           |

Keys can also be passed directly via `api_keys={"openrouter": "sk-or-..."}`.

> **Note:** Models must support structured output (JSON schema via `response_format`). Most recent OpenAI and OpenRouter models do; check your provider's documentation for specifics.

---

## Web UI

A browser-based interface is available on [Hugging Face Spaces](https://huggingface.co/spaces/karimelhajjami/synthline) or self-hosted with Docker.

```bash
git clone https://github.com/abdelkarim-elhajjami/Synthline.git && cd Synthline && ./dev.sh
```

---

## Project Structure

```
synthline/          SDK package (pip install synthline)
  core/             FM parser, resolver, generator, PACE, alignment verifier
  utils/            Logger, parsing, progress tracking
  client.py         Synthline class — build_prompts(), optimize(), generate()
  types.py          PromptSet, Dataset
  cli.py            CLI entry point
server/             FastAPI + WebSocket server for the Web UI
tests/              Unit and integration tests
web/                Next.js frontend
```

## Citation

```bibtex
@software{synthline,
  author = {El Hajjami, Abdelkarim},
  title = {Synthline: Feature Model–Guided Synthetic Data Generator},
  url = {https://github.com/abdelkarim-elhajjami/Synthline},
  year = {2025},
}
```

## License

[Apache License 2.0](https://github.com/abdelkarim-elhajjami/Synthline/blob/main/LICENSE)