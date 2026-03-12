"""Tests for the shared-base verification workflow (verify + Dataset persistence)."""

import asyncio
import json
import os
import tempfile

import pytest
from unittest.mock import AsyncMock, MagicMock

from synthline.client import Synthline
from synthline.types import Dataset, GenerationResult


# ======================================================================
# Helpers
# ======================================================================


def _make_config(config_id: str) -> dict:
    return {
        "llm": "test-model",
        "temperature": 0.7,
        "top_p": 1.0,
        "__fm_constraints__": [{"label": f"Feat_{config_id}", "value": config_id}],
        "classification_label": "Test",
        "classification_label_def": "A test label",
        "prompt": f"Generate data for config {config_id}",
        # Top-level feature keys (as real configs have — format_sample uses these)
        f"Feat_{config_id}": config_id,
    }


def _make_raw_sample(text: str, config: dict) -> dict:
    return {"text": text, "config": config}


def _scored(sample, score):
    return (sample, score)


def _make_dataset_with_metadata(samples_data, config_id="A"):
    """Build a Dataset with formatted samples and prompts metadata."""
    config = _make_config(config_id)
    formatted = [
        Dataset.format_sample(s["text"], s["config"])
        for s in samples_data
    ]
    features = {k: v for k, v in formatted[0].items() if k not in ("Text", "Label")} if formatted else {}
    prompts_meta = [{
        "prompt": config["prompt"],
        "features": features,
        "samples_produced": len(formatted),
        "optimized": False,
    }]
    metadata = {
        "run_id": "test-run",
        "samples_per_prompt": 20,
        "prompts": prompts_meta,
    }
    return Dataset(samples=formatted, metadata=metadata)


class _FakeClient:
    """Minimal stand-in for Synthline to call verify directly."""

    def __init__(self, verifier, generator, api_keys=None):
        self._runtime = MagicMock()
        self._runtime.align_verifier = verifier
        self._runtime.generator = generator
        self._llm = "test-model"
        self._temperature = 0.7
        self._top_p = 1.0
        self._api_keys = api_keys or {}

    async def run_verify(self, dataset, threshold=0.5, samples_per_prompt=20):
        return await Synthline.verify(self, dataset, threshold=threshold, samples_per_prompt=samples_per_prompt)


# ======================================================================
# Dataset persistence tests
# ======================================================================


class TestDatasetPersistence:
    def test_save_writes_csv_and_metadata(self):
        """Dataset.save() writes data.csv and metadata.json (no raw_samples.json)."""
        ds = Dataset(
            samples=[{"Text": "hello", "Label": "Test"}],
            metadata={"run_id": "test-123"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "data.csv"))
            assert os.path.exists(os.path.join(tmpdir, "metadata.json"))
            assert not os.path.exists(os.path.join(tmpdir, "raw_samples.json"))

    def test_load_round_trips(self):
        """Dataset.load() restores samples and metadata."""
        original = Dataset(
            samples=[{"Text": "sample text", "Label": "Test"}],
            metadata={"run_id": "rt-789", "samples_per_prompt": 20},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            original.save(tmpdir)
            loaded = Dataset.load(tmpdir)

            assert len(loaded.samples) == 1
            assert loaded.samples[0]["Text"] == "sample text"
            assert loaded.metadata["run_id"] == "rt-789"
            assert loaded.metadata["samples_per_prompt"] == 20


# ======================================================================
# unformat_sample tests
# ======================================================================


class TestUnformatSample:
    def test_round_trip(self):
        """format_sample → unformat_sample preserves text, label, and features."""
        config = _make_config("A")
        formatted = Dataset.format_sample("hello world", config)
        raw = Dataset.unformat_sample(formatted, prompt_text="Generate data for config A")

        assert raw["text"] == "hello world"
        assert raw["config"]["classification_label"] == "Test"
        assert raw["config"]["prompt"] == "Generate data for config A"
        constraints = raw["config"]["__fm_constraints__"]
        labels = {c["label"] for c in constraints}
        assert "Feat_A" in labels

    def test_without_prompt(self):
        """unformat_sample works without prompt text."""
        formatted = {"Text": "hi", "Label": "X", "Domain": "D1"}
        raw = Dataset.unformat_sample(formatted)

        assert raw["text"] == "hi"
        assert raw["config"]["classification_label"] == "X"
        assert "prompt" not in raw["config"]
        assert raw["config"]["Domain"] == "D1"

    def test_features_as_top_level_config_keys(self):
        """Feature columns are preserved as top-level config keys."""
        formatted = {"Text": "t", "Label": "L", "Domain": "D", "Context": "C"}
        raw = Dataset.unformat_sample(formatted)

        assert raw["config"]["Domain"] == "D"
        assert raw["config"]["Context"] == "C"
        constraints = {c["label"]: c["value"] for c in raw["config"]["__fm_constraints__"]}
        assert constraints == {"Domain": "D", "Context": "C"}


# ======================================================================
# verify tests
# ======================================================================


class TestVerify:
    def test_verify_all_accepted(self):
        """All samples pass verification → returned as-is."""

        async def run():
            config = _make_config("A")
            raw = [
                _make_raw_sample("s1", config),
                _make_raw_sample("s2", config),
            ]
            ds = _make_dataset_with_metadata(raw, "A")

            reconstructed = [
                Dataset.unformat_sample(s, config["prompt"])
                for s in ds.samples
            ]

            verifier = MagicMock()
            verifier.verify.return_value = (
                [_scored(s, 0.9) for s in reconstructed],
                [],
            )
            generator = AsyncMock()
            client = _FakeClient(verifier, generator)

            dataset = await client.run_verify(ds, threshold=0.6)

            assert len(dataset.samples) == 2
            assert dataset.metadata["verify"] is True
            assert dataset.metadata["source_run_id"] == "test-run"
            assert dataset.metadata["alignment_verification"]["termination_reason"] == "count_reached"
            generator.generate_for_configs.assert_not_called()

        asyncio.run(run())

    def test_verify_with_regeneration(self):
        """Some samples rejected → regenerated from same config."""

        async def run():
            config = _make_config("A")
            raw = [
                _make_raw_sample("good", config),
                _make_raw_sample("bad", config),
            ]
            ds = _make_dataset_with_metadata(raw, "A")

            recon_good = Dataset.unformat_sample(ds.samples[0], config["prompt"])
            recon_bad = Dataset.unformat_sample(ds.samples[1], config["prompt"])
            regen = _make_raw_sample("regen", config)

            verifier = MagicMock()
            verifier.verify.side_effect = [
                ([_scored(recon_good, 0.8)], [_scored(recon_bad, 0.3)]),
                ([_scored(regen, 0.9)], []),
            ]

            generator = AsyncMock()
            generator.generate_for_configs.return_value = GenerationResult(
                samples=[regen]
            )

            client = _FakeClient(verifier, generator)
            dataset = await client.run_verify(ds, threshold=0.6)

            assert len(dataset.samples) == 2
            generator.generate_for_configs.assert_called_once()
            call_kwargs = generator.generate_for_configs.call_args.kwargs
            assert call_kwargs["samples_per_prompt"] == 20

        asyncio.run(run())

    def test_verify_preserves_original_metadata(self):
        """Original metadata fields are preserved in the output."""

        async def run():
            config = _make_config("A")
            raw = [_make_raw_sample("s1", config)]
            ds = _make_dataset_with_metadata(raw, "A")
            ds.metadata["llm"] = "test-model"
            ds.metadata["optimized"] = False

            reconstructed = Dataset.unformat_sample(ds.samples[0], config["prompt"])

            verifier = MagicMock()
            verifier.verify.return_value = ([_scored(reconstructed, 0.9)], [])
            generator = AsyncMock()
            client = _FakeClient(verifier, generator)

            dataset = await client.run_verify(ds, threshold=0.6)

            assert dataset.metadata["llm"] == "test-model"
            assert dataset.metadata["optimized"] is False
            assert dataset.metadata["verify"] is True
            assert dataset.metadata["source_run_id"] == "test-run"
            assert dataset.metadata["run_id"] != "test-run"

        asyncio.run(run())

    def test_verify_full_round_trip(self):
        """Generate → save → load → verify → save → load: end-to-end."""

        async def run():
            config = _make_config("A")
            raw = [
                _make_raw_sample("good", config),
                _make_raw_sample("bad", config),
            ]

            c1 = _make_dataset_with_metadata(raw, "A")
            c1.metadata["run_id"] = "c1-run"

            with tempfile.TemporaryDirectory() as tmpdir:
                c1_path = os.path.join(tmpdir, "c1")
                c1.save(c1_path)

                loaded_c1 = Dataset.load(c1_path)
                assert len(loaded_c1.samples) == 2

                recon_good = Dataset.unformat_sample(loaded_c1.samples[0], config["prompt"])
                recon_bad = Dataset.unformat_sample(loaded_c1.samples[1], config["prompt"])
                regen = _make_raw_sample("regen", config)

                verifier = MagicMock()
                verifier.verify.side_effect = [
                    ([_scored(recon_good, 0.8)], [_scored(recon_bad, 0.3)]),
                    ([_scored(regen, 0.9)], []),
                ]
                generator = AsyncMock()
                generator.generate_for_configs.return_value = GenerationResult(
                    samples=[regen]
                )
                client = _FakeClient(verifier, generator)

                c3 = await client.run_verify(loaded_c1, threshold=0.6)

                c3_path = os.path.join(tmpdir, "c3")
                c3.save(c3_path)

                loaded_c3 = Dataset.load(c3_path)
                assert len(loaded_c3.samples) == 2
                assert loaded_c3.metadata["source_run_id"] == "c1-run"
                assert not os.path.exists(os.path.join(c3_path, "raw_samples.json"))

        asyncio.run(run())

    def test_verify_empty_dataset(self):
        """verify() handles empty datasets gracefully."""

        async def run():
            ds = Dataset(
                samples=[],
                metadata={"samples_per_prompt": 1, "prompts": []},
            )

            verifier = MagicMock()
            verifier.verify.return_value = ([], [])
            generator = AsyncMock()
            client = _FakeClient(verifier, generator)

            dataset = await client.run_verify(ds, threshold=0.6)
            assert len(dataset.samples) == 0

        asyncio.run(run())

    def test_verify_after_csv_round_trip(self):
        """save → load → verify works correctly despite CSV string coercion."""

        async def run():
            config = _make_config("A")
            raw = [
                _make_raw_sample("s1", config),
                _make_raw_sample("s2", config),
            ]
            ds = _make_dataset_with_metadata(raw, "A")

            with tempfile.TemporaryDirectory() as tmpdir:
                ds.save(tmpdir)
                loaded = Dataset.load(tmpdir)

                # After CSV round-trip, all values are strings
                assert all(isinstance(v, str) for v in loaded.samples[0].values())

                reconstructed = [
                    Dataset.unformat_sample(s, config["prompt"])
                    for s in loaded.samples
                ]

                verifier = MagicMock()
                verifier.verify.return_value = (
                    [_scored(s, 0.9) for s in reconstructed],
                    [],
                )
                generator = AsyncMock()
                client = _FakeClient(verifier, generator)

                dataset = await client.run_verify(loaded, threshold=0.6)

                assert len(dataset.samples) == 2
                assert dataset.metadata["verify"] is True
                # Prompt lookup should still work after string coercion
                assert dataset.metadata["alignment_verification"]["termination_reason"] == "count_reached"

        asyncio.run(run())
