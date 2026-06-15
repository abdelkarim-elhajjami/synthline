import pytest
from unittest.mock import AsyncMock, MagicMock

from synthline.core.pace import PACE
from synthline.errors import (
    StructuredOutputCompatibilityError,
    StructuredOutputError,
    StructuredOutputResponseError,
)


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm._max_concurrency = 10
    return llm


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
        assert results[0].prompt == "Write a user story for login including acceptance criteria"
        assert isinstance(results[0].score, float)
        assert results[0].score > 0.0
        assert mock_llm.get_batch_completions.call_count >= 4

    asyncio.run(run())


def test_optimize_batch_normalizes_string_sample_count(pace_instance):
    import asyncio

    async def run():
        pace_instance._optimize_atomic_prompt = AsyncMock(return_value=("prompt", 0.5))

        await pace_instance.optimize_batch(
            atomic_configs=[{"prompt": "Initial prompt"}],
            features={"samples_per_prompt": "2", "pace_alpha": 0.0},
            n_iterations=1,
            n_actors=1,
            n_candidates=1,
        )

        sent_features = pace_instance._optimize_atomic_prompt.call_args.kwargs["features"]
        assert sent_features["samples_per_prompt"] == 2

    asyncio.run(run())


def test_evaluate_prompt_logic(pace_instance):
    pace_instance._diversity_score = MagicMock(return_value=0.65)

    features = {"pace_alpha": 0.0}
    raw_completion = '{"samples": ["Requirement 1", "Requirement 2"]}'
    score = pace_instance._evaluate_prompt([raw_completion], samples_per_prompt=2, features=features)
    assert score == pytest.approx(0.65)

    with pytest.raises(StructuredOutputError, match="invalid JSON"):
        pace_instance._evaluate_prompt(
            ["Invalid JSON"], samples_per_prompt=2, features=features
        )


def test_update_prompt_handles_failure(pace_instance, mock_llm, mock_logger):
    import asyncio

    async def run():
        mock_llm.get_batch_completions.side_effect = Exception("Update failed")

        with pytest.raises(Exception, match="Update failed"):
            await pace_instance._update_prompt("Initial Prompt", ["Feedback"], {"llm": "gpt-4", "temperature": 1.0, "top_p": 1.0}, initial_prompt="Initial Prompt")
        mock_logger.log_error.assert_called()

    asyncio.run(run())


def test_optimize_batch_raises_when_a_config_fails(pace_instance):
    import asyncio

    async def run():
        pace_instance._optimize_atomic_prompt = AsyncMock(
            side_effect=[
                RuntimeError("Provider failed"),
                ("prompt_ok", 0.2),
            ]
        )

        atomic_configs = [{"prompt": "p1"}, {"prompt": "p2"}]
        features = {"samples_per_prompt": 1, "llm": "gpt-4"}

        with pytest.raises(RuntimeError, match="Provider failed"):
            await pace_instance.optimize_batch(
                atomic_configs=atomic_configs,
                features=features,
                n_iterations=1,
                n_actors=1,
                n_candidates=1,
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_iterations": 0, "n_actors": 1, "n_candidates": 1}, "n_iterations"),
        ({"n_iterations": 1, "n_actors": 0, "n_candidates": 1}, "n_actors"),
        ({"n_iterations": 1, "n_actors": 1, "n_candidates": 0}, "n_candidates"),
    ],
)
def test_optimize_batch_rejects_invalid_work_counts(pace_instance, kwargs, message):
    import asyncio

    async def run():
        with pytest.raises(ValueError, match=message):
            await pace_instance.optimize_batch(
                atomic_configs=[{"prompt": "p1"}],
                features={"samples_per_prompt": 1, "pace_alpha": 0.0},
                **kwargs,
            )

    asyncio.run(run())


def test_optimize_batch_rejects_empty_configs(pace_instance):
    import asyncio

    async def run():
        with pytest.raises(ValueError, match="at least one atomic configuration"):
            await pace_instance.optimize_batch(
                atomic_configs=[],
                features={"samples_per_prompt": 1, "pace_alpha": 0.0},
                n_iterations=1,
                n_actors=1,
                n_candidates=1,
            )

    asyncio.run(run())


def test_actor_reports_schema_violation_as_response_error(pace_instance, mock_llm):
    import asyncio

    async def run():
        mock_llm.get_batch_completions.return_value = ["plain text"]

        with pytest.raises(
            StructuredOutputResponseError,
            match="response may have been truncated",
        ):
            await pace_instance._run_actor(
                "Generate a requirement",
                {
                    "llm": "ollama/test-model",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "samples_per_prompt": 1,
                },
            )
        assert mock_llm.get_batch_completions.call_count == 1

    asyncio.run(run())


def test_actor_preserves_llm_structured_response_error(pace_instance, mock_llm):
    import asyncio

    async def run():
        error = StructuredOutputResponseError("Provider returned no completion choice.")
        mock_llm.get_batch_completions.side_effect = error

        with pytest.raises(StructuredOutputResponseError) as exc_info:
            await pace_instance._run_actor(
                "Generate a requirement",
                {
                    "llm": "openai/gpt-4.1-mini",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "samples_per_prompt": 1,
                },
            )

        assert exc_info.value is error

    asyncio.run(run())


def test_prompt_update_callback_failure_aborts_optimization(pace_instance):
    import asyncio

    async def run():
        pace_instance._run_actor = AsyncMock(return_value='{"samples": ["A", "B"]}')
        pace_instance._run_critic = AsyncMock(return_value="Improve clarity")
        pace_instance._update_prompt = AsyncMock(return_value="Improved prompt")
        pace_instance._evaluate_prompt = MagicMock(return_value=0.8)
        callback = AsyncMock(side_effect=RuntimeError("WebSocket send failed"))

        with pytest.raises(RuntimeError, match="WebSocket send failed"):
            await pace_instance._optimize_atomic_prompt(
                features={"samples_per_prompt": 2, "pace_alpha": 0.0},
                n_iterations=1,
                n_actors=1,
                n_candidates=1,
                initial_prompt="Initial prompt",
                prompt_update_callback=callback,
                atomic_config_index=0,
                total_configs=1,
            )

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
