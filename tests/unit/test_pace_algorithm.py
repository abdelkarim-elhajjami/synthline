import pytest
from unittest.mock import AsyncMock, MagicMock

from synthline.core.pace import PACE


@pytest.fixture
def mock_llm():
    return AsyncMock()


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def pace_instance(mock_llm, mock_logger):
    return PACE(mock_llm, mock_logger)


def test_optimize_batch_flow(pace_instance, mock_llm):
    """Test the high-level batch optimization flow."""
    import asyncio

    async def run():
        atomic_configs = [{"prompt": "Write a user story for login", "classification_label": "Security"}]
        features = {
            "domain": "Cybersecurity",
            "classification_label": "Security",
            "classification_label_def": "A security constraint",
            "language": "English",
            "stakeholder_viewpoint": "Security Engineer",
            "requirement_type": "Quality",
            "requirement_subtype": "Security",
            "context": "Technical",
            "abstraction_level": "DetailedLevel",
            "samples_per_prompt": 2,
            "pace_alpha": 0.0,
            "llm": "gpt-4",
            "temperature": 1.0,
            "top_p": 1.0,
        }

        actor_response = '{"samples": ["As a user, I want 2FA enabled to secure my account.", "As an admin, I want to enforce password complexity."]}'
        critic_response = "Critique: The stories lack acceptance criteria."
        update_response = "Write a user story for login including acceptance criteria"

        mock_llm.get_batch_completions.side_effect = [
            [actor_response],
            [critic_response],
            [update_response],
            [actor_response],
        ]

        results = await pace_instance.optimize_batch(
            atomic_configs,
            features,
            n_iterations=1,
            n_actors=1,
            n_candidates=1,
        )

        assert len(results) == 1
        best_prompt, best_score, _ = results[0]
        assert best_prompt == "Write a user story for login including acceptance criteria"
        assert isinstance(best_score, float)
        assert best_score > 0.0
        assert mock_llm.get_batch_completions.call_count >= 4

    asyncio.run(run())


def test_evaluate_prompt_logic(pace_instance):
    pace_instance._diversity_score = MagicMock(return_value=0.65)

    features = {"pace_alpha": 0.0}
    raw_completion = '{"samples": ["Requirement 1", "Requirement 2"]}'
    score = pace_instance._evaluate_prompt([raw_completion], samples_per_prompt=2, features=features)
    assert score == pytest.approx(0.65)

    score_invalid = pace_instance._evaluate_prompt(
        ["Invalid JSON"], samples_per_prompt=2, features=features
    )
    assert score_invalid == 0.0


def test_update_prompt_handles_failure(pace_instance, mock_llm, mock_logger):
    import asyncio

    async def run():
        mock_llm.get_batch_completions.side_effect = Exception("Update failed")

        with pytest.raises(Exception, match="Update failed"):
            await pace_instance._update_prompt("Initial Prompt", ["Feedback"], {"llm": "gpt-4", "temperature": 1.0, "top_p": 1.0}, initial_prompt="Initial Prompt")
        mock_logger.log_error.assert_called()

    asyncio.run(run())


def test_optimize_batch_degrades_gracefully_on_llm_error(pace_instance, mock_logger):
    import asyncio

    async def run():
        pace_instance._optimize_atomic_prompt = AsyncMock(
            side_effect=[
                RuntimeError("Provider failed"),
                ("optimized_p2", 0.8),
            ]
        )

        atomic_configs = [{"prompt": "p1"}, {"prompt": "p2"}]
        features = {"samples_per_prompt": 1, "llm": "gpt-4"}

        results = await pace_instance.optimize_batch(
            atomic_configs=atomic_configs,
            features=features,
            n_iterations=1,
            n_actors=1,
            n_candidates=1,
        )

        assert len(results) == 2

        prompt_0, score_0, _ = results[0]
        assert prompt_0 == "p1"
        assert score_0 == 0.0

        prompt_1, score_1, _ = results[1]
        assert prompt_1 == "optimized_p2"
        assert score_1 == 0.8

        mock_logger.log_warning.assert_called_once()

    asyncio.run(run())


def test_evaluate_prompt_uses_weighted_diversity_and_alignment(mock_llm, mock_logger):
    align_scorer = MagicMock()
    align_scorer.score_batch.return_value = 0.8
    pace = PACE(mock_llm, mock_logger, align_scorer=align_scorer)
    pace._diversity_score = MagicMock(return_value=0.6)

    raw_completion = '{"samples": ["A short requirement", "A different requirement"]}'
    features = {
        "pace_alpha": 0.25,
        "__fm_constraints__": [{"label": "Language", "value": "English"}],
    }

    score = pace._evaluate_prompt([raw_completion], samples_per_prompt=2, features=features)

    assert score == pytest.approx((0.25 * 0.8) + (0.75 * 0.6))
    align_scorer.score_batch.assert_called_once()
    pace._diversity_score.assert_called_once()


def test_evaluate_prompt_requires_align_scorer_for_non_diversity_only(mock_llm, mock_logger):
    pace = PACE(mock_llm, mock_logger)
    features = {
        "pace_alpha": 0.5,
        "__fm_constraints__": [{"label": "Language", "value": "English"}],
    }

    with pytest.raises(ValueError, match="requires AlignScorer"):
        pace._evaluate_prompt(
            raw_completions=['{"samples": ["Requirement A", "Requirement B"]}'],
            samples_per_prompt=2,
            features=features,
        )


def test_evaluate_prompt_skips_alignment_when_alpha_zero(mock_llm, mock_logger):
    align_scorer = MagicMock()
    align_scorer.score_batch = MagicMock(side_effect=AssertionError("Alignment should not be computed"))
    pace = PACE(mock_llm, mock_logger, align_scorer=align_scorer)
    pace._diversity_score = MagicMock(return_value=0.61)

    features = {
        "pace_alpha": 0.0,
        "__fm_constraints__": [{"label": "Language", "value": "English"}],
    }

    score = pace._evaluate_prompt(
        raw_completions=['{"samples": ["Requirement A", "Requirement B"]}'],
        samples_per_prompt=2,
        features=features,
    )

    assert score == pytest.approx(0.61)
    align_scorer.score_batch.assert_not_called()


def test_evaluate_prompt_skips_diversity_when_alpha_one(mock_llm, mock_logger):
    align_scorer = MagicMock()
    align_scorer.score_batch.return_value = 0.74
    pace = PACE(mock_llm, mock_logger, align_scorer=align_scorer)
    pace._diversity_score = MagicMock(side_effect=AssertionError("Diversity should not be computed"))

    features = {
        "pace_alpha": 1.0,
        "__fm_constraints__": [{"label": "Language", "value": "English"}],
    }

    score = pace._evaluate_prompt(
        raw_completions=['{"samples": ["Requirement A", "Requirement B"]}'],
        samples_per_prompt=2,
        features=features,
    )

    assert score == pytest.approx(0.74)
    align_scorer.score_batch.assert_called_once()
