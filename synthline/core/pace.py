"""
Implementation of PACE (Prompt Actor-Critic Editing) for Synthline.
https://aclanthology.org/2024.findings-acl.436/
"""
from typing import Any, Awaitable, Callable, Dict, List, Tuple, Optional
import asyncio
from synthline.core.align_scorer import AlignScorer
from synthline.core.constants import extract_fm_constraints
from synthline.core.llm import LLMClient
from synthline.utils.logger import Logger
from synthline.core.schemas import samples_schema
from synthline.utils.parsing import parse_completion
from synthline.types import PromptUpdateCallback
from synthline.utils.progress import ProgressFn, track_progress


class PACE:
    def __init__(
        self,
        llm_client: LLMClient,
        logger: Logger,
        align_scorer: Optional[AlignScorer] = None,
    ) -> None:
        self._llm = llm_client
        self._logger = logger
        self._align_scorer = align_scorer
        self._sbert_model = None
        self._sbert_load_attempted = False

    def set_align_scorer(self, align_scorer: Optional[AlignScorer]) -> None:
        """Configure the optional alignment scorer used during prompt evaluation."""
        self._align_scorer = align_scorer

    async def optimize_batch(
        self,
        atomic_configs: List[Dict[str, Any]],
        features: Dict[str, Any],
        *,
        n_iterations: int,
        n_actors: int,
        n_candidates: int,
        progress_callback: ProgressFn = None,
        prompt_update_callback: PromptUpdateCallback = None,
        api_keys: Optional[Dict[str, str]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Optimize multiple prompts in parallel (one for each atomic configuration)."""
        tasks: List[asyncio.Task] = []
        total_configs = len(atomic_configs)

        completed_iterations = 0
        total_iterations = total_configs * n_iterations

        async def update_progress():
            nonlocal completed_iterations
            completed_iterations += 1
            progress = (completed_iterations / total_iterations) * 100
            if progress_callback:
                await track_progress(progress_callback, progress)

        async def _run_config(
            config_idx: int,
            atomic_config: Dict[str, Any],
        ) -> Tuple[int, str, float, Dict[str, Any]]:
            features_merged = {**features, **atomic_config}
            initial_prompt = atomic_config.get('prompt', None)
            try:
                prompt, score = await self._optimize_atomic_prompt(
                    features=features_merged,
                    progress_callback=update_progress,
                    initial_prompt=initial_prompt,
                    n_iterations=n_iterations,
                    n_actors=n_actors,
                    n_candidates=n_candidates,
                    prompt_update_callback=prompt_update_callback,
                    atomic_config_index=config_idx,
                    total_configs=total_configs,
                    api_keys=api_keys,
                )
            except Exception as e:
                self._logger.log_warning(
                    f"PACE config {config_idx} failed, "
                    f"falling back to original prompt: {e}",
                    "pace_batch",
                )
                prompt = initial_prompt or ""
                score = 0.0
            return config_idx, prompt, score, atomic_config

        for i, atomic_config in enumerate(atomic_configs):
            tasks.append(asyncio.create_task(_run_config(i, atomic_config)))

        indexed_results: Dict[int, Tuple[str, float, Dict[str, Any]]] = {}
        for task in asyncio.as_completed(tasks):
            idx, prompt, score, atomic_config = await task
            indexed_results[idx] = (prompt, score, atomic_config)

        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for idx in sorted(indexed_results):
            prompt, score, atomic_config = indexed_results[idx]
            results.append((prompt, score, atomic_config))

        # Ensure progress reaches 100% when finished
        if progress_callback:
            await track_progress(progress_callback, 100)

        return results

    async def _optimize_atomic_prompt(
        self,
        *,
        features: Dict[str, Any],
        n_iterations: int,
        n_actors: int,
        n_candidates: int,
        progress_callback: Optional[Callable[[], Awaitable[None]]] = None,
        initial_prompt: Optional[str] = None,
        prompt_update_callback: PromptUpdateCallback = None,
        atomic_config_index: Optional[int] = None,
        total_configs: Optional[int] = None,
        api_keys: Optional[Dict[str, str]] = None
    ) -> Tuple[str, float]:
        """Optimize a single atomic prompt."""
        current_prompt = initial_prompt
        best_prompt = current_prompt
        best_score = 0.0

        # Repeat until convergence or max iterations
        for t in range(n_iterations):
            all_critiques = []
            all_actions = []

            for _ in range(n_actors):
                action = await self._run_actor(prompt=current_prompt, features=features, api_keys=api_keys)
                all_actions.append(action)

                critique = await self._run_critic(
                    prompt=current_prompt,
                    action=action,
                    features=features,
                    api_keys=api_keys,
                )
                all_critiques.append(critique)

            all_candidate_prompts = []
            all_candidate_scores = []

            for _ in range(n_candidates):
                candidate_prompt = await self._update_prompt(current_prompt, all_critiques, features, initial_prompt=initial_prompt, api_keys=api_keys)

                new_actions = []
                for _ in range(n_actors):
                    action = await self._run_actor(prompt=candidate_prompt, features=features, api_keys=api_keys)
                    new_actions.append(action)

                candidate_score = self._evaluate_prompt(
                    raw_completions=new_actions,
                    samples_per_prompt=features["samples_per_prompt"],
                    features=features,
                )

                all_candidate_prompts.append(candidate_prompt)
                all_candidate_scores.append(candidate_score)

            if all_candidate_prompts:
                best_idx = all_candidate_scores.index(max(all_candidate_scores))
                candidate_prompt = all_candidate_prompts[best_idx]
                candidate_score = all_candidate_scores[best_idx]

                if candidate_score > best_score:
                    best_prompt = candidate_prompt
                    best_score = candidate_score

                    self._logger.log_prompt(
                        prompt=best_prompt,
                        score=best_score,
                        event="NEW BEST PROMPT",
                        config=features
                    )

                current_prompt = candidate_prompt

                if prompt_update_callback:
                    try:
                        await prompt_update_callback(
                            best_prompt,
                            best_score,
                            t + 1,
                            n_iterations,
                            atomic_config_index,
                            total_configs,
                        )
                    except Exception as cb_error:
                        self._logger.log_error(
                            f"Prompt update callback error: {str(cb_error)}",
                            "pace",
                            {"iteration": t + 1}
                        )

            if progress_callback:
                await progress_callback()

        # Log final results
        self._logger.log_prompt(
            prompt=best_prompt,
            score=best_score,
            event="FINAL OPTIMIZED PROMPT",
            config=features
        )

        return best_prompt, best_score

    async def _run_actor(
        self,
        prompt: str,
        features: Dict[str, Any],
        api_keys: Optional[Dict[str, str]] = None
    ) -> str:
        """Run the actor to generate synthetic samples based on the current prompt."""
        try:
            spp = max(1, int(features.get("samples_per_prompt", 1)))
            completions = await self._llm.get_batch_completions(
                prompts=[prompt],
                features=features,
                api_keys=api_keys,
                response_format=samples_schema(spp),
            )
            return completions[0]

        except Exception as e:
            self._logger.log_error(
                f"Actor error: {str(e)}",
                "pace",
                {"prompt": prompt},
            )
            raise

    async def _run_critic(
        self,
        prompt: str,
        action: str,
        features: Dict[str, Any],
        api_keys: Optional[Dict[str, str]] = None
    ) -> str:
        """Run the critic to provide a critique with suggestions for refining the prompt."""
        constraints_text = self._build_constraints_text(features)
        samples_per_prompt = features.get("samples_per_prompt", 1)

        critique_prompt = f"""Instruction:
"{prompt}"

Output ({samples_per_prompt} items):
{action}

Verify that:
1. Each item satisfies:
{constraints_text}
2. The items are semantically diverse to support downstream classifier generalization.

If the output falls short on any point, describe specifically how."""

        critic_settings = {
            'llm': features['llm'],
            'temperature': features['temperature'],
            'top_p': features['top_p'],
        }

        try:
            completions = await self._llm.get_batch_completions(
                prompts=[critique_prompt],
                features=critic_settings,
                api_keys=api_keys
            )
            return completions[0] if completions else ""

        except Exception as e:
            self._logger.log_error(
                f"Critic error: {str(e)}",
                "pace",
                {"prompt": prompt, "action": action},
            )
            raise

    async def _update_prompt(
        self,
        current_prompt: str,
        feedback_list: List[str],
        features: Dict[str, Any],
        initial_prompt: str,
        api_keys: Optional[Dict[str, str]] = None
    ) -> str:
        """Update the prompt based on collected feedback, ensuring format instructions are preserved."""

        combined_feedback = "\n\n".join([f"Feedback {i+1}:\n{fb}" for i, fb in enumerate(feedback_list)])
        update_prompt = f"""Current Instruction:
"{current_prompt}"

Reference Instruction:
"{initial_prompt}"

Critiques:
{combined_feedback}

Rewrite the Current Instruction to address the Critiques, preserving from the Reference Instruction:
- The sample count and artefact type
- All constraints exactly as stated

Return only the rewritten instruction."""

        update_settings = {
            'llm': features['llm'],
            'temperature': features['temperature'],
            'top_p': features['top_p'],
        }

        try:
            completions = await self._llm.get_batch_completions(
                prompts=[update_prompt],
                features=update_settings,
                api_keys=api_keys
            )
            updated_prompt = completions[0].strip() if completions and completions[0] else current_prompt
            # Strip surrounding quotes if the LLM added them
            if len(updated_prompt) >= 2 and updated_prompt.startswith('"') and updated_prompt.endswith('"'):
                updated_prompt = updated_prompt[1:-1].strip()
            return updated_prompt

        except Exception as e:
            self._logger.log_error(
                f"Update error: {str(e)}",
                "pace",
                {"current_prompt": current_prompt},
            )
            raise

    def _build_constraints_text(self, features: Dict[str, Any]) -> str:
        """Build constraint text from explicit FM constraints only."""
        lines = []
        for label, value in extract_fm_constraints(features):
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value if str(v).strip())
            else:
                value_str = str(value)
            if not value_str.strip():
                continue
            lines.append(f"- Satisfy {label}: {value_str}.")
        return "\n".join(lines)

    def _evaluate_prompt(
        self,
        raw_completions: List[str],
        samples_per_prompt: int,
        features: Dict[str, Any],
    ) -> float:
        """Evaluate the prompt using weighted diversity and alignment scores.
        Returns 0.0 if any completion fails to parse correctly or yields the wrong number of samples.
        """
        parsed_samples = []

        for raw_completion in raw_completions:
            parsed = parse_completion(raw_completion, samples_per_prompt)
            # Strict check 1: Did parsing fail completely?
            if parsed is None:
                return 0.0

            # Strict check 2: Did parsing yield the exact number of expected samples?
            if len(parsed) != samples_per_prompt:
                return 0.0

            # If checks pass, add the valid samples
            parsed_samples.extend(parsed)

        # Need at least two samples to calculate pairwise distance
        if len(parsed_samples) <= 1:
            return 0.0

        alpha = self._resolve_alpha(features)
        if alpha == 0.0:
            return self._diversity_score(parsed_samples)

        if self._align_scorer is None:
            raise ValueError(
                "PACE requires AlignScorer when pace_alpha is greater than 0.0."
            )

        try:
            alignment_score = self._align_scorer.score_batch(
                samples=parsed_samples,
                attributes=features,
            )
        except Exception as e:
            self._logger.log_error(
                f"Alignment scoring failed: {str(e)}",
                "pace_eval",
            )
            raise

        if alpha == 1.0:
            return alignment_score

        diversity_score = self._diversity_score(parsed_samples)
        return (alpha * alignment_score) + ((1.0 - alpha) * diversity_score)

    def _resolve_alpha(self, features: Dict[str, Any]) -> float:
        raw_alpha = features.get("pace_alpha", 0.5)
        try:
            alpha = float(raw_alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid pace_alpha value: {raw_alpha}") from exc

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"pace_alpha must be between 0.0 and 1.0. Received: {alpha}")
        return alpha

    def _diversity_score(self, parsed_samples: List[str]) -> float:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_distances

        model = self._get_sbert_model()
        embeddings = model.encode(parsed_samples)
        distances = cosine_distances(embeddings)
        avg_distance = np.mean(distances[np.triu_indices(len(parsed_samples), k=1)])
        return float(avg_distance)

    def _get_sbert_model(self):
        if self._sbert_load_attempted:
            if self._sbert_model is None:
                raise RuntimeError("SBERT model is not available after initialization failure.")
            return self._sbert_model

        self._sbert_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            self._sbert_model = SentenceTransformer("all-mpnet-base-v2")
            return self._sbert_model
        except Exception as exc:
            self._sbert_model = None
            raise RuntimeError(
                f"Failed to initialize SBERT model for diversity scoring: {exc}"
            ) from exc
