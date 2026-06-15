from __future__ import annotations

from typing import Union
from unittest.mock import MagicMock, patch

import pytest

from synthline.core.align_verifier import AlignVerifier
from synthline.errors import AlignmentVerificationError


def _make_sample(text: str, constraints=None):
    config = {}
    if constraints is not None:
        config["__fm_constraints__"] = constraints
    return {"text": text, "config": config}


def _make_verifier(scores_map: Union[dict, list]):
    """Create an AlignVerifier with a mocked AlignScorer.

    scores_map can be:
      - a list of floats: returned for every score_samples call
      - a dict mapping text -> score for precise control
    """
    scorer = MagicMock()

    if isinstance(scores_map, list):
        scorer.score_samples.return_value = scores_map
    else:
        def _score_samples(texts, attributes):
            return [scores_map.get(t, 0.0) for t in texts]
        scorer.score_samples.side_effect = _score_samples

    logger = MagicMock()
    return AlignVerifier(align_scorer=scorer, logger=logger)


class TestVerifyAcceptsAboveThreshold:
    def test_all_above(self):
        samples = [
            _make_sample("good one"),
            _make_sample("also good"),
        ]
        verifier = _make_verifier([0.9, 0.85])

        accepted, rejected = verifier.verify(samples, threshold=0.7)

        assert len(accepted) == 2
        assert len(rejected) == 0

    def test_all_accepted_contain_original_samples(self):
        samples = [_make_sample("text A"), _make_sample("text B")]
        verifier = _make_verifier([0.8, 0.9])

        accepted, _ = verifier.verify(samples, threshold=0.5)

        accepted_texts = {s[0]["text"] for s in accepted}
        assert accepted_texts == {"text A", "text B"}


class TestVerifyRejectsBelowThreshold:
    def test_split_correctly(self):
        samples = [
            _make_sample("high"),
            _make_sample("low"),
            _make_sample("mid"),
        ]
        verifier = _make_verifier({"high": 0.92, "low": 0.3, "mid": 0.75})

        accepted, rejected = verifier.verify(samples, threshold=0.7)

        assert len(accepted) == 2
        assert len(rejected) == 1
        assert rejected[0][0]["text"] == "low"

    def test_all_rejected(self):
        samples = [_make_sample("bad A"), _make_sample("bad B")]
        verifier = _make_verifier({"bad A": 0.1, "bad B": 0.2})

        accepted, rejected = verifier.verify(samples, threshold=0.5)

        assert len(accepted) == 0
        assert len(rejected) == 2


class TestVerifyAcceptedSortedByScore:
    def test_best_first_ordering(self):
        samples = [
            _make_sample("low"),
            _make_sample("high"),
            _make_sample("mid"),
        ]
        verifier = _make_verifier({"low": 0.71, "high": 0.99, "mid": 0.85})

        accepted, _ = verifier.verify(samples, threshold=0.7)

        scores = [s[1] for s in accepted]
        assert scores == sorted(scores, reverse=True)
        assert accepted[0][0]["text"] == "high"
        assert accepted[1][0]["text"] == "mid"
        assert accepted[2][0]["text"] == "low"


class TestVerifyEmptySamples:
    def test_returns_empty_tuples(self):
        verifier = _make_verifier([])

        accepted, rejected = verifier.verify([], threshold=0.5)

        assert accepted == []
        assert rejected == []


class TestVerifyEdgeCases:
    def test_threshold_boundary_exact_match_accepted(self):
        samples = [_make_sample("exact")]
        verifier = _make_verifier({"exact": 0.7})

        accepted, rejected = verifier.verify(samples, threshold=0.7)

        assert len(accepted) == 1
        assert len(rejected) == 0

    def test_threshold_boundary_just_below_rejected(self):
        samples = [_make_sample("close")]
        verifier = _make_verifier({"close": 0.6999})

        accepted, rejected = verifier.verify(samples, threshold=0.7)

        assert len(accepted) == 0
        assert len(rejected) == 1

    def test_scorer_failure_aborts_verification(self):
        scorer = MagicMock()
        scorer.score_samples.side_effect = RuntimeError("model failed")
        logger = MagicMock()
        verifier = AlignVerifier(align_scorer=scorer, logger=logger)

        samples = [_make_sample("will fail")]
        with pytest.raises(
            AlignmentVerificationError,
            match="could not score the generated samples",
        ):
            verifier.verify(samples, threshold=0.5)

    def test_scorer_result_count_must_match_samples(self):
        scorer = MagicMock()
        scorer.score_samples.return_value = []
        verifier = AlignVerifier(align_scorer=scorer, logger=MagicMock())

        with pytest.raises(
            AlignmentVerificationError,
            match="unexpected number of scores",
        ):
            verifier.verify([_make_sample("missing score")], threshold=0.5)

    def test_max_retries_constant(self):
        assert AlignVerifier.MAX_RETRIES == 3
