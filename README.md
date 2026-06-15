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
- **Use compatible LLMs** through OpenAI, OpenRouter, Ollama (local), or Hugging Face Inference API.
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
output = await sl.generate(prompts=prompts, samples=1000)

# 3. Export
output.save("output/")       # data.csv, metadata.json
df = output.to_dataframe()   # pandas DataFrame
```

### With PACE Optimization

```python
optimized = await sl.optimize(prompts, alpha=0.5, iterations=1, actors=4, candidates=2)
output = await sl.generate(prompts=optimized, samples=1000)
```

### With Alignment Verification

```python
output = await sl.generate(prompts=prompts, samples=1000, verify=True, verify_threshold=0.6)
```

All SDK methods are async except `build_prompts` (no LLM call).

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
For Ollama, set `OLLAMA_BASE_URL` to the server root, such as
`http://localhost:11434`; Synthline automatically uses its OpenAI-compatible `/v1` API.

> **Reasoning models are not supported.** Synthline is designed for predictable,
> high-throughput synthetic data sampling. Reasoning models are generally slower and costlier,
> and many reject the `temperature` and `top_p` controls Synthline uses. Select a standard chat
> or instruct model; known reasoning families and explicit reasoning options fail immediately
> with an actionable error.

> **Required:** The selected LLM must support **strict structured outputs with JSON Schema**
> (`response_format.type = "json_schema"`). JSON mode alone is not sufficient. Synthline does
> not fall back to plaintext or suppress provider/schema errors. Choosing a compatible model is
> the user's responsibility.

Alignment verification also fails loudly if its NLI scorer cannot run. Infrastructure failures
are never treated as low-scoring samples.

Compatibility is determined by the **model + provider + endpoint** combination. The same model
can support structured outputs through one serving stack and not through another.

The Web UI lists only OpenRouter models that advertise strict structured outputs and every
standard sampling parameter Synthline sends, and removes known reasoning-model families.
Generation requests require OpenRouter to route only through providers that support the
requested parameters. OpenAI, Hugging Face, Ollama, and other model catalogs do not expose an
equally reliable per-model capability flag through the endpoints Synthline uses, so verify those
models against the provider documentation before selecting them:

- [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs)
- [OpenAI structured outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Mistral structured outputs](https://docs.mistral.ai/studio-api/conversations/structured-output)

---

## Web UI

A browser-based interface is available on [Hugging Face Spaces](https://huggingface.co/spaces/abdelkarim-elhajjami/synthline) or self-hosted with Docker.

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
