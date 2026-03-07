from unittest.mock import MagicMock, patch

import pytest

from synthline.core.align_scorer import AlignScorer


def test_build_attribute_hypothesis_from_fm_constraints():
    align = AlignScorer(
        logger=MagicMock(),
        model_name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    )
    attributes = {
        "__fm_constraints__": [
            {"label": "Language", "value": "English"},
            {"label": "Stakeholder", "value": ["Patient", "Clinician"]},
        ]
    }

    hypothesis = align._get_cached_hypothesis(attributes)

    assert "Language is English" in hypothesis
    assert "Stakeholder is Patient, Clinician" in hypothesis


def test_build_attribute_hypothesis_rejects_missing_constraints():
    align = AlignScorer(
        logger=MagicMock(),
        model_name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    )

    with pytest.raises(ValueError, match="requires non-empty constraints"):
        align._get_cached_hypothesis({})


def test_default_model_and_entailment_index_are_explicit():
    assert AlignScorer.DEFAULT_MODEL_NAME == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    assert AlignScorer.ENTAILMENT_CLASS_INDEX == 0


def test_hypothesis_cache_key_is_order_stable():
    align = AlignScorer(
        logger=MagicMock(),
        model_name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    )

    attrs_a = {
        "__fm_constraints__": [
            {"label": "Language", "value": "English"},
            {"label": "Stakeholder", "value": ["Patient", "Clinician"]},
        ]
    }
    attrs_b = {
        "__fm_constraints__": [
            {"label": "Stakeholder", "value": ["Patient", "Clinician"]},
            {"label": "Language", "value": "English"},
        ]
    }

    stmts_a = align._validate_and_extract_constraint_fragments(attrs_a)
    stmts_b = align._validate_and_extract_constraint_fragments(attrs_b)
    key_a = align._build_hypothesis_cache_key(stmts_a)
    key_b = align._build_hypothesis_cache_key(stmts_b)
    assert key_a == key_b


def test_score_samples_returns_individual_scores():
    """score_samples() must return a list with one float per sample."""
    import numpy as np

    align = AlignScorer(
        logger=MagicMock(),
        model_name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    )

    fake_logits_np = np.array([[2.0, 0.5, 0.1], [0.3, 1.8, 0.2]])

    class FakeOutputs:
        pass

    with patch.object(align, "_get_runtime_components") as mock_components:
        import torch

        fake_logits = torch.tensor(fake_logits_np, dtype=torch.float32)
        fake_outputs = FakeOutputs()
        fake_outputs.logits = fake_logits

        mock_model = MagicMock(return_value=fake_outputs)
        mock_tokenizer = MagicMock(return_value={"input_ids": torch.zeros(2, 10, dtype=torch.long)})
        mock_components.return_value = (mock_model, mock_tokenizer, 0)

        attributes = {
            "__fm_constraints__": [{"label": "Language", "value": "English"}]
        }
        scores = align.score_samples(["Sample A", "Sample B"], attributes)

        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        # First sample has high logit at index 0 → high entailment prob
        assert scores[0] > scores[1]


def test_score_batch_delegates_to_score_samples():
    """score_batch() should return the mean of score_samples()."""
    align = AlignScorer(
        logger=MagicMock(),
        model_name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    )
    fake_scores = [0.8, 0.6, 0.4]

    with patch.object(align, "score_samples", return_value=fake_scores) as mock_ss:
        attributes = {
            "__fm_constraints__": [{"label": "Tone", "value": "Formal"}]
        }
        result = align.score_batch(["a", "b", "c"], attributes)

        mock_ss.assert_called_once_with(["a", "b", "c"], attributes)
        assert abs(result - 0.6) < 1e-9


def test_score_samples_rejects_empty_batch():
    align = AlignScorer(
        logger=MagicMock(),
        model_name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    )
    with pytest.raises(ValueError, match="empty sample batch"):
        align.score_samples([], {})

