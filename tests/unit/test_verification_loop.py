"""Tests for the config-aware verification engine."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from synthline.core.verification import (
    config_key,
    run_verification_loop,
    _group_rejected_by_config,
)
from synthline.errors import AlignmentVerificationError
from synthline.types import GenerationResult


# ======================================================================
# Helpers
# ======================================================================


def _make_config(config_id: str, *, optimized: bool = False) -> dict:
    """Distinct config per ID — mirrors what generator.py produces."""
    config = {
        "llm": "test-model",
        "temperature": 0.7,
        "top_p": 1.0,
        "__fm_constraints__": [{"label": f"Feat_{config_id}", "value": config_id}],
        "classification_label": "Test",
        "classification_label_def": "A test label",
        "prompt": f"Generate data for config {config_id}",
    }
    if optimized:
        config["optimized_prompt"] = f"Optimized prompt for config {config_id}"
    return config


def _make_sample(text: str, config: dict) -> dict:
    return {"text": text, "config": config}


def _scored(sample: dict, score: float):
    return (sample, score)


async def _run_loop(verifier, generator, raw_samples, features, threshold, samples_needed):
    return await run_verification_loop(
        raw_samples=raw_samples,
        features=features,
        threshold=threshold,
        samples_needed=samples_needed,
        verifier=verifier,
        generator=generator,
    )


# ======================================================================
# _config_key tests
# ======================================================================


class TestConfigKey:
    def test_excludes_prompt(self):
        config = {"llm": "model", "prompt": "hello", "temperature": 0.7}
        key = config_key(config)
        assert "prompt" not in key
        assert "llm" in key

    def test_same_config_same_key(self):
        c1 = {"llm": "model", "temperature": 0.7, "prompt": "a"}
        c2 = {"llm": "model", "temperature": 0.7, "prompt": "b"}
        assert config_key(c1) == config_key(c2)

    def test_different_config_different_key(self):
        c1 = {"llm": "model-a", "temperature": 0.7, "prompt": "x"}
        c2 = {"llm": "model-b", "temperature": 0.7, "prompt": "x"}
        assert config_key(c1) != config_key(c2)

    def test_excludes_optimized_prompt_and_pace_score(self):
        c1 = {"llm": "model", "temperature": 0.7, "prompt": "a"}
        c2 = {
            "llm": "model",
            "temperature": 0.7,
            "prompt": "a",
            "optimized_prompt": "better a",
            "pace_score": 0.85,
        }
        assert config_key(c1) == config_key(c2)


# ======================================================================
# _group_rejected_by_config tests
# ======================================================================


class TestGroupRejectedByConfig:
    def test_single_config_multiple_rejections(self):
        config = _make_config("A")
        rejected = [
            _scored(_make_sample("s1", config), 0.3),
            _scored(_make_sample("s2", config), 0.4),
        ]
        groups = _group_rejected_by_config(rejected)
        assert len(groups) == 1
        _, count = list(groups.values())[0]
        assert count == 2

    def test_multiple_configs(self):
        config_a = _make_config("A")
        config_b = _make_config("B")
        rejected = [
            _scored(_make_sample("a1", config_a), 0.3),
            _scored(_make_sample("b1", config_b), 0.2),
            _scored(_make_sample("a2", config_a), 0.4),
        ]
        groups = _group_rejected_by_config(rejected)
        assert len(groups) == 2

        key_a = config_key(config_a)
        key_b = config_key(config_b)
        assert groups[key_a][1] == 2
        assert groups[key_b][1] == 1

    def test_empty_rejected(self):
        groups = _group_rejected_by_config([])
        assert groups == {}


# ======================================================================
# _run_verification_loop tests
# ======================================================================


class TestVerificationLoop:
    """Tests for the config-aware verification loop."""

    def _features(self, spp=20):
        return {"samples_per_prompt": spp, "llm": "test", "temperature": 0.7, "top_p": 1.0}

    def test_all_accepted_first_attempt(self):
        """All samples pass → no regeneration, count_reached."""

        async def run():
            config = _make_config("A")
            samples = [_make_sample(f"s{i}", config) for i in range(3)]

            verifier = MagicMock()
            verifier.verify.return_value = (
                [_scored(s, 0.9) for s in samples],  # all accepted
                [],  # no rejected
            )

            generator = AsyncMock()

            accepted, meta, warnings = await _run_loop(verifier, generator,
                samples, self._features(), threshold=0.6, samples_needed=3
            )

            assert len(accepted) == 3
            assert meta["termination_reason"] == "count_reached"
            generator.generate_for_configs.assert_not_called()

        asyncio.run(run())

    def test_verifier_infrastructure_failure_aborts_without_regeneration(self):
        async def run():
            config = _make_config("A")
            sample = _make_sample("sample", config)
            verifier = MagicMock()
            verifier.verify.side_effect = AlignmentVerificationError("NLI unavailable")
            generator = AsyncMock()

            with pytest.raises(AlignmentVerificationError, match="NLI unavailable"):
                await _run_loop(
                    verifier,
                    generator,
                    [sample],
                    self._features(),
                    threshold=0.6,
                    samples_needed=1,
                )

            generator.generate_for_configs.assert_not_called()

        asyncio.run(run())

    def test_config_aware_regen_single_config(self):
        """Rejected samples from one config → regeneration targets that config."""

        async def run():
            config = _make_config("A")
            s1 = _make_sample("good", config)
            s2 = _make_sample("bad", config)

            verifier = MagicMock()
            # Attempt 1: 1 accepted, 1 rejected
            # Attempt 2: 1 accepted (from regen)
            verifier.verify.side_effect = [
                ([_scored(s1, 0.8)], [_scored(s2, 0.3)]),
                ([_scored(_make_sample("regen1", config), 0.9)], []),
            ]

            regen_result = GenerationResult(
                samples=[_make_sample("regen1", config)],
            )
            generator = AsyncMock()
            generator.generate_for_configs.return_value = regen_result

            accepted, meta, warnings = await _run_loop(verifier, generator,
                [s1, s2], self._features(spp=20), threshold=0.6, samples_needed=2
            )

            assert len(accepted) == 2
            # Verify generate_for_configs was called with the right config
            call_args = generator.generate_for_configs.call_args
            config_requests = call_args.kwargs["config_requests"]
            assert len(config_requests) == 1
            req_config, req_count = config_requests[0]
            assert req_config == config
            assert req_count == 1  # 1 rejected → 1 needed

        asyncio.run(run())

    def test_config_aware_regen_multi_config(self):
        """Two configs, different rejection counts → both targeted."""

        async def run():
            config_a = _make_config("A")
            config_b = _make_config("B")

            accepted_samples = [_make_sample("a_ok", config_a)]
            rejected_samples = [
                _make_sample("a_bad", config_a),
                _make_sample("b_bad1", config_b),
                _make_sample("b_bad2", config_b),
            ]

            verifier = MagicMock()
            verifier.verify.side_effect = [
                (
                    [_scored(s, 0.8) for s in accepted_samples],
                    [_scored(s, 0.3) for s in rejected_samples],
                ),
                (
                    [
                        _scored(_make_sample("a_regen", config_a), 0.9),
                        _scored(_make_sample("b_regen1", config_b), 0.8),
                        _scored(_make_sample("b_regen2", config_b), 0.7),
                    ],
                    [],
                ),
            ]

            regen_samples = [
                _make_sample("a_regen", config_a),
                _make_sample("b_regen1", config_b),
                _make_sample("b_regen2", config_b),
            ]
            generator = AsyncMock()
            generator.generate_for_configs.return_value = GenerationResult(
                samples=regen_samples
            )

            accepted, meta, warnings = await _run_loop(verifier, generator,
                accepted_samples + rejected_samples,
                self._features(spp=20),
                threshold=0.6,
                samples_needed=4,
            )

            assert len(accepted) == 4
            # Check config_requests
            call_args = generator.generate_for_configs.call_args
            config_requests = call_args.kwargs["config_requests"]
            counts = {config_key(c): n for c, n in config_requests}
            assert counts[config_key(config_a)] == 1
            assert counts[config_key(config_b)] == 2

        asyncio.run(run())

    def test_max_retries_exhausted(self):
        """Samples keep failing → terminates with max_retries_reached."""

        async def run():
            config = _make_config("A")
            bad = _make_sample("bad", config)

            verifier = MagicMock()
            # All 4 attempts: always rejected
            verifier.verify.return_value = ([], [_scored(bad, 0.2)])

            generator = AsyncMock()
            generator.generate_for_configs.return_value = GenerationResult(
                samples=[_make_sample("still_bad", config)]
            )

            accepted, meta, warnings = await _run_loop(verifier, generator,
                [bad], self._features(spp=20), threshold=0.6, samples_needed=1
            )

            assert len(accepted) == 0
            assert meta["termination_reason"] == "max_retries_reached"
            assert meta["alignment_deficit"] == 1
            assert "verification_deficit" in warnings
            # 3 regeneration calls (attempts 1→2, 2→3, 3→4; attempt 4 is final)
            assert generator.generate_for_configs.call_count == 3

        asyncio.run(run())

    def test_deficit_capped_by_global_need(self):
        """5 rejected across configs, but only 2 more samples needed globally."""

        async def run():
            config_a = _make_config("A")
            config_b = _make_config("B")

            # 8 accepted, 5 rejected, but we only need 10 total
            accepted_samples = [_make_sample(f"ok{i}", config_a) for i in range(8)]
            rejected_samples = [
                _make_sample("a_bad1", config_a),
                _make_sample("a_bad2", config_a),
                _make_sample("a_bad3", config_a),
                _make_sample("b_bad1", config_b),
                _make_sample("b_bad2", config_b),
            ]

            verifier = MagicMock()
            verifier.verify.side_effect = [
                (
                    [_scored(s, 0.9) for s in accepted_samples],
                    [_scored(s, 0.2) for s in rejected_samples],
                ),
                (
                    [
                        _scored(_make_sample("regen1", config_a), 0.8),
                        _scored(_make_sample("regen2", config_a), 0.8),
                    ],
                    [],
                ),
            ]

            generator = AsyncMock()
            generator.generate_for_configs.return_value = GenerationResult(
                samples=[
                    _make_sample("regen1", config_a),
                    _make_sample("regen2", config_a),
                ]
            )

            accepted, meta, warnings = await _run_loop(verifier, generator,
                accepted_samples + rejected_samples,
                self._features(spp=20),
                threshold=0.6,
                samples_needed=10,
            )

            assert len(accepted) == 10
            # Deficit was 2, not 5 — config_requests should total at most 2
            call_args = generator.generate_for_configs.call_args
            config_requests = call_args.kwargs["config_requests"]
            total_requested = sum(c for _, c in config_requests)
            assert total_requested == 2

        asyncio.run(run())

    def test_empty_regeneration_terminates(self):
        """Generator returns no samples → terminates."""

        async def run():
            config = _make_config("A")
            bad = _make_sample("bad", config)

            verifier = MagicMock()
            verifier.verify.return_value = ([], [_scored(bad, 0.2)])

            generator = AsyncMock()
            generator.generate_for_configs.return_value = GenerationResult(samples=[])

            accepted, meta, warnings = await _run_loop(verifier, generator,
                [bad], self._features(spp=20), threshold=0.6, samples_needed=1
            )

            assert len(accepted) == 0
            assert meta["termination_reason"] == "generation_returned_empty"

        asyncio.run(run())

    def test_prompt_preserved_in_regen_request(self):
        """The config passed to generate_for_configs includes the original prompt."""

        async def run():
            config = _make_config("A")
            bad = _make_sample("bad", config)

            verifier = MagicMock()
            verifier.verify.side_effect = [
                ([], [_scored(bad, 0.2)]),
                ([_scored(_make_sample("good", config), 0.9)], []),
            ]

            generator = AsyncMock()
            generator.generate_for_configs.return_value = GenerationResult(
                samples=[_make_sample("good", config)]
            )

            await _run_loop(verifier, generator,
                [bad], self._features(spp=20), threshold=0.6, samples_needed=1
            )

            call_args = generator.generate_for_configs.call_args
            config_requests = call_args.kwargs["config_requests"]
            req_config, _ = config_requests[0]
            # The original prompt is preserved in the config
            assert req_config["prompt"] == "Generate data for config A"

        asyncio.run(run())

    def test_optimized_prompt_preserved(self):
        """PACE scenario: optimized_prompt flows through to regeneration."""

        async def run():
            config = _make_config("A", optimized=True)
            bad = _make_sample("bad", config)

            verifier = MagicMock()
            verifier.verify.side_effect = [
                ([], [_scored(bad, 0.2)]),
                ([_scored(_make_sample("good", config), 0.9)], []),
            ]

            generator = AsyncMock()
            generator.generate_for_configs.return_value = GenerationResult(
                samples=[_make_sample("good", config)]
            )

            await _run_loop(verifier, generator,
                [bad], self._features(spp=20), threshold=0.6, samples_needed=1
            )

            call_args = generator.generate_for_configs.call_args
            config_requests = call_args.kwargs["config_requests"]
            req_config, _ = config_requests[0]
            assert req_config.get("optimized_prompt") == "Optimized prompt for config A"

        asyncio.run(run())

    def test_metadata_structure(self):
        """Verify alignment_verification metadata has expected keys."""

        async def run():
            config = _make_config("A")
            samples = [_make_sample("s1", config)]

            verifier = MagicMock()
            verifier.verify.return_value = (
                [_scored(samples[0], 0.9)],
                [],
            )

            generator = AsyncMock()
            _, meta, _ = await _run_loop(verifier, generator,
                samples, self._features(), threshold=0.6, samples_needed=1
            )

            assert "alignment_threshold" in meta
            assert "max_retries" in meta
            assert "attempts_used" in meta
            assert "requested_samples" in meta
            assert "accepted_samples" in meta
            assert "alignment_deficit" in meta
            assert "termination_reason" in meta
            assert "attempt_trace" in meta
            assert "total_generated_across_retries" in meta

        asyncio.run(run())

    def test_attempt_trace_includes_rejected_configs(self):
        """attempt_trace should include rejected_configs count."""

        async def run():
            config_a = _make_config("A")
            config_b = _make_config("B")
            samples = [
                _make_sample("a1", config_a),
                _make_sample("b1", config_b),
            ]

            verifier = MagicMock()
            verifier.verify.return_value = (
                [],
                [_scored(s, 0.2) for s in samples],
            )

            generator = AsyncMock()
            generator.generate_for_configs.return_value = GenerationResult(samples=[])

            _, meta, _ = await _run_loop(verifier, generator,
                samples, self._features(spp=20), threshold=0.6, samples_needed=2
            )

            # First attempt should show 2 rejected configs
            trace = meta["attempt_trace"][0]
            assert trace["rejected_configs"] == 2

        asyncio.run(run())
