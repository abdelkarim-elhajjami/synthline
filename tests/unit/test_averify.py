"""Tests for the shared-base verification workflow (averify + Dataset persistence)."""

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
    }


def _make_raw_sample(text: str, config: dict) -> dict:
    return {"text": text, "config": config}


def _scored(sample, score):
    return (sample, score)


class _FakeClient:
    """Minimal stand-in for Synthline to call averify directly."""

    def __init__(self, verifier, generator, api_keys=None):
        self._runtime = MagicMock()
        self._runtime.align_verifier = verifier
        self._runtime.generator = generator
        self._llm = "test-model"
        self._temperature = 0.7
        self._top_p = 1.0
        self._api_keys = api_keys or {}

    async def run_averify(self, dataset, threshold=0.5):
        # Bind both averify and _run_verification_loop from Synthline
        return await Synthline.averify(self, dataset, threshold=threshold)

    async def _run_verification_loop(self, *args, **kwargs):
        return await Synthline._run_verification_loop(self, *args, **kwargs)


# ======================================================================
# Dataset persistence tests
# ======================================================================


class TestDatasetPersistence:
    def test_save_writes_raw_samples_json(self):
        """Dataset.save() writes raw_samples.json when raw_samples is present."""
        config = _make_config("A")
        raw = [_make_raw_sample("hello", config)]
        ds = Dataset(
            samples=[{"Text": "hello", "Label": "Test"}],
            metadata={"run_id": "test-123"},
            raw_samples=raw,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "raw_samples.json"))
            assert os.path.exists(os.path.join(tmpdir, "data.csv"))
            assert os.path.exists(os.path.join(tmpdir, "metadata.json"))

            loaded = json.loads(open(os.path.join(tmpdir, "raw_samples.json")).read())
            assert len(loaded) == 1
            assert loaded[0]["text"] == "hello"

    def test_save_skips_raw_samples_when_none(self):
        """Dataset.save() does not write raw_samples.json when raw_samples is None."""
        ds = Dataset(
            samples=[{"Text": "hello"}],
            metadata={"run_id": "test-456"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.save(tmpdir)
            assert not os.path.exists(os.path.join(tmpdir, "raw_samples.json"))
            assert os.path.exists(os.path.join(tmpdir, "data.csv"))

    def test_load_round_trips_all_files(self):
        """Dataset.load() restores samples, metadata, and raw_samples."""
        config = _make_config("A")
        raw = [_make_raw_sample("sample text", config)]
        original = Dataset(
            samples=[{"Text": "sample text", "Label": "Test"}],
            metadata={"run_id": "rt-789", "samples_per_prompt": 20},
            raw_samples=raw,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            original.save(tmpdir)
            loaded = Dataset.load(tmpdir)

            assert len(loaded.samples) == 1
            assert loaded.samples[0]["Text"] == "sample text"
            assert loaded.metadata["run_id"] == "rt-789"
            assert loaded.metadata["samples_per_prompt"] == 20
            assert loaded.raw_samples is not None
            assert len(loaded.raw_samples) == 1
            assert loaded.raw_samples[0]["text"] == "sample text"

    def test_load_handles_missing_raw_samples(self):
        """Dataset.load() returns raw_samples=None for old datasets."""
        ds = Dataset(
            samples=[{"Text": "old"}],
            metadata={"run_id": "old-1"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.save(tmpdir)
            loaded = Dataset.load(tmpdir)

            assert loaded.raw_samples is None
            assert len(loaded.samples) == 1


# ======================================================================
# averify tests
# ======================================================================


class TestAverify:
    def test_averify_raises_without_raw_samples(self):
        """averify() raises ValueError when raw_samples is None."""
        ds = Dataset(samples=[], metadata={})

        verifier = MagicMock()
        generator = AsyncMock()
        client = _FakeClient(verifier, generator)

        with pytest.raises(ValueError, match="no raw_samples"):
            asyncio.run(client.run_averify(ds))

    def test_averify_all_accepted(self):
        """All samples pass verification → returned as-is."""

        async def run():
            config = _make_config("A")
            raw = [
                _make_raw_sample("s1", config),
                _make_raw_sample("s2", config),
            ]
            ds = Dataset(
                samples=[],
                metadata={"run_id": "src-1", "samples_per_prompt": 20},
                raw_samples=raw,
            )

            verifier = MagicMock()
            verifier.verify.return_value = (
                [_scored(s, 0.9) for s in raw],
                [],
            )
            generator = AsyncMock()
            client = _FakeClient(verifier, generator)

            result = await client.run_averify(ds, threshold=0.6)

            assert len(result.samples) == 2
            assert result.metadata["verify"] is True
            assert result.metadata["source_run_id"] == "src-1"
            assert result.metadata["alignment_verification"]["termination_reason"] == "count_reached"
            generator.generate_for_configs.assert_not_called()

        asyncio.run(run())

    def test_averify_with_regeneration(self):
        """Some samples rejected → regenerated from same config."""

        async def run():
            config = _make_config("A")
            good = _make_raw_sample("good", config)
            bad = _make_raw_sample("bad", config)
            regen = _make_raw_sample("regen", config)

            ds = Dataset(
                samples=[],
                metadata={"run_id": "src-2", "samples_per_prompt": 20},
                raw_samples=[good, bad],
            )

            verifier = MagicMock()
            verifier.verify.side_effect = [
                ([_scored(good, 0.8)], [_scored(bad, 0.3)]),
                ([_scored(regen, 0.9)], []),
            ]

            generator = AsyncMock()
            generator.generate_for_configs.return_value = GenerationResult(
                samples=[regen]
            )

            client = _FakeClient(verifier, generator)
            result = await client.run_averify(ds, threshold=0.6)

            assert len(result.samples) == 2
            generator.generate_for_configs.assert_called_once()
            # spp should come from metadata
            call_kwargs = generator.generate_for_configs.call_args.kwargs
            assert call_kwargs["samples_per_prompt"] == 20

        asyncio.run(run())

    def test_averify_preserves_original_metadata(self):
        """Original metadata fields are preserved in the output."""

        async def run():
            config = _make_config("A")
            raw = [_make_raw_sample("s1", config)]
            ds = Dataset(
                samples=[],
                metadata={
                    "run_id": "original-run",
                    "llm": "test-model",
                    "samples_per_prompt": 20,
                    "optimized": False,
                },
                raw_samples=raw,
            )

            verifier = MagicMock()
            verifier.verify.return_value = ([_scored(raw[0], 0.9)], [])
            generator = AsyncMock()
            client = _FakeClient(verifier, generator)

            result = await client.run_averify(ds, threshold=0.6)

            # Original fields preserved
            assert result.metadata["llm"] == "test-model"
            assert result.metadata["optimized"] is False
            # New fields added
            assert result.metadata["verify"] is True
            assert result.metadata["source_run_id"] == "original-run"
            # run_id is new (not the original)
            assert result.metadata["run_id"] != "original-run"

        asyncio.run(run())

    def test_averify_returns_raw_samples(self):
        """averify() populates raw_samples on the returned Dataset."""

        async def run():
            config = _make_config("A")
            raw = [_make_raw_sample("s1", config)]
            ds = Dataset(
                samples=[],
                metadata={"samples_per_prompt": 1},
                raw_samples=raw,
            )

            verifier = MagicMock()
            verifier.verify.return_value = ([_scored(raw[0], 0.9)], [])
            generator = AsyncMock()
            client = _FakeClient(verifier, generator)

            result = await client.run_averify(ds, threshold=0.6)

            assert result.raw_samples is not None
            assert len(result.raw_samples) == 1
            assert result.raw_samples[0]["text"] == "s1"

        asyncio.run(run())

    def test_averify_full_round_trip(self):
        """Generate → save → load → verify → save → load: end-to-end."""

        async def run():
            config = _make_config("A")
            raw = [
                _make_raw_sample("good", config),
                _make_raw_sample("bad", config),
            ]

            # Simulate C1 dataset
            c1 = Dataset(
                samples=[{"Text": "good", "Label": "Test"}, {"Text": "bad", "Label": "Test"}],
                metadata={"run_id": "c1-run", "samples_per_prompt": 20},
                raw_samples=raw,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                c1_path = os.path.join(tmpdir, "c1")
                c1.save(c1_path)

                # Load C1
                loaded_c1 = Dataset.load(c1_path)
                assert loaded_c1.raw_samples is not None
                assert len(loaded_c1.raw_samples) == 2

                # Verify to produce C3
                regen = _make_raw_sample("regen", config)
                verifier = MagicMock()
                verifier.verify.side_effect = [
                    ([_scored(raw[0], 0.8)], [_scored(raw[1], 0.3)]),
                    ([_scored(regen, 0.9)], []),
                ]
                generator = AsyncMock()
                generator.generate_for_configs.return_value = GenerationResult(
                    samples=[regen]
                )
                client = _FakeClient(verifier, generator)

                c3 = await client.run_averify(loaded_c1, threshold=0.6)

                # Save C3
                c3_path = os.path.join(tmpdir, "c3")
                c3.save(c3_path)

                # Load C3 and verify
                loaded_c3 = Dataset.load(c3_path)
                assert len(loaded_c3.samples) == 2
                assert loaded_c3.metadata["source_run_id"] == "c1-run"
                assert loaded_c3.raw_samples is not None

        asyncio.run(run())
