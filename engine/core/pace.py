"""
Implementation of PACE (Prompt Actor-Critic Editing) for Synthline.
https://aclanthology.org/2024.findings-acl.436/
"""
from typing import Any, Dict, List, Tuple, Optional
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from core.llm import LLMClient
from utils.logger import Logger
from utils.parsing import parse_completion
from utils.progress import ProgressCallback, track_progress
from utils.ctx import SystemContext

class PACE:
    """Implements the PACE approach for prompt optimization."""
    def __init__(
        self,
        llm_client: LLMClient,
        logger: Logger,
    ) -> None:
        """Initialize the PACE optimizer."""
        self._llm = llm_client
        self._logger = logger
        self._model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def optimize_batch(
        self,
        atomic_configs: List[Dict[str, Any]],
        features: Dict[str, Any],
        progress_callback: ProgressCallback = None,
        n_iterations: Optional[int] = None,
        n_actors: Optional[int] = None,
        n_candidates: Optional[int] = None,
        system_ctx: Optional[SystemContext] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Optimize multiple prompts in parallel (one for each atomic configuration).
        """
        tasks = []
        total_configs = len(atomic_configs)
        
        completed_iterations = 0
        total_iterations = total_configs * n_iterations
        
        # Function to update progress when an iteration completes
        async def update_progress():
            nonlocal completed_iterations
            completed_iterations += 1
            progress = (completed_iterations / total_iterations) * 100
            if progress_callback:
                await track_progress(progress_callback, progress)
        
        # Create optimization tasks for each atomic config
        for i, atomic_config in enumerate(atomic_configs):
            # Create a merged feature dict with both base and atomic config
            features_merged = {**features, **atomic_config}
            
            # Generate initial prompt for this config if not provided
            initial_prompt = atomic_config.get('prompt', None)
            
            # Create a task for optimizing this prompt
            task = asyncio.create_task(
                self._optimize_atomic_prompt(
                    features=features_merged,
                    progress_callback=update_progress,
                    initial_prompt=initial_prompt,
                    n_iterations=n_iterations,
                    n_actors=n_actors,
                    n_candidates=n_candidates,
                    system_ctx=system_ctx,
                    atomic_config_index=i,
                    total_configs=total_configs
                )
            )
            tasks.append((task, atomic_config))
        
        results = []
        
        # Wait for all tasks to complete and collect results
        for task, atomic_config in tasks:
            try:
                prompt, score = await task
                results.append((prompt, score, atomic_config))
            except Exception as e:
                self._logger.log_error(
                    f"Optimization failed for atomic_config: {str(e)}", 
                    "pace_batch", 
                    {"atomic_config": str(atomic_config)}
                )
                # Add empty result for failed config
                results.append(("", 0.0, atomic_config))
        
        # Ensure progress reaches 100% when finished
        if progress_callback:
            await track_progress(progress_callback, 100)
        
        # Send final batch results via WebSocket if available
        if system_ctx and features.get('connection_id'):
            websocket = system_ctx.get_connection(features['connection_id'])
            if websocket:
                try:
                    await websocket.send_json({
                        "type": "optimize_complete_batch",
                        "optimized_results": [
                            {
                                "prompt": prompt,
                                "score": float(score),
                                "atomic_config": {
                                    k: v for k, v in atomic_config.items()
                                    if k in ['label', 'label_definition', 'specification_format', 
                                           'specification_level', 'stakeholder', 'domain', 
                                           'language', 'samples_per_prompt']
                                }
                            } for prompt, score, atomic_config in results
                        ]
                    })
                except Exception as ws_error:
                    self._logger.log_error(
                        f"Failed to send batch completion: {str(ws_error)}", 
                        "websocket"
                    )
        
        return results
    
    async def _optimize_atomic_prompt(
        self,
        features: Dict[str, Any],
        progress_callback: ProgressCallback = None,
        initial_prompt: Optional[str] = None,
        n_iterations: Optional[int] = None,
        n_actors: Optional[int] = None,
        n_candidates: Optional[int] = None,
        system_ctx: Optional[SystemContext] = None,
        atomic_config_index: Optional[int] = None,
        total_configs: Optional[int] = None
    ) -> Tuple[str, float]:
        """Optimize a single atomic prompt using PACE."""        
        # Initialization:
        current_prompt = initial_prompt
        best_prompt = current_prompt
        best_score = 0.0
        
        try:
            # Repeat until convergence or max iterations
            for t in range(n_iterations):
                # 1. Collect feedback using n actors and critics (only once per iteration)
                all_critiques = []
                all_actions = []
                
                for _ in range(n_actors):
                    try:
                        action = await self._run_actor(prompt=current_prompt, features=features)
                        all_actions.append(action)
                        
                        critique = await self._run_critic(prompt=current_prompt, action=action, features=features)
                        all_critiques.append(critique)
                        
                    except Exception as e:
                        self._logger.log_error(f"Actor-critic error: {str(e)}", "pace", {"prompt": current_prompt})
                
                # 2. Generate multiple candidate prompts from the SAME set of critiques
                all_candidate_prompts = []
                all_candidate_scores = []
                
                for _ in range(n_candidates):
                    try:
                        # 3. Update the prompt based on critiques (generate one candidate)
                        candidate_prompt = await self._update_prompt(current_prompt, all_critiques, features)
                        
                        # 4. Evaluate each candidate prompt
                        new_actions = []
                        for _ in range(n_actors):
                            action = await self._run_actor(prompt=candidate_prompt, features=features)
                            new_actions.append(action)
                        
                        candidate_score = self._evaluate_prompt(new_actions, features['samples_per_prompt'])
                        
                        all_candidate_prompts.append(candidate_prompt)
                        all_candidate_scores.append(candidate_score)
                        
                    except Exception as e:
                        self._logger.log_error(f"Candidate generation error: {str(e)}", "pace", {"prompt": current_prompt})
                
                # 5. Select the best candidate from this iteration
                if all_candidate_prompts:
                    best_idx = all_candidate_scores.index(max(all_candidate_scores))
                    candidate_prompt = all_candidate_prompts[best_idx]
                    candidate_score = all_candidate_scores[best_idx]
                    
                    # Update the best prompt if this candidate is better
                    if candidate_score > best_score:
                        best_prompt = candidate_prompt
                        best_score = candidate_score
                        
                        # Log when we find a better prompt
                        self._logger.log_prompt(
                            prompt=best_prompt,
                            score=best_score,
                            event="NEW BEST PROMPT",
                            config=features
                        )
                    
                    # Update current prompt for next iteration
                    current_prompt = candidate_prompt
                    
                    # Send update via WebSocket if connection exists
                    if system_ctx and features.get('connection_id'):
                        websocket = system_ctx.get_connection(features['connection_id'])
                        if websocket:
                            try:
                                await websocket.send_json({
                                    "type": "prompt_update",
                                    "prompt": best_prompt,
                                    "score": best_score,
                                    "iteration": t + 1,
                                    "atomic_config_index": atomic_config_index,
                                    "total_configs": total_configs
                                })
                            except Exception as ws_error:
                                self._logger.log_error(
                                    f"WebSocket send error: {str(ws_error)}", 
                                    "pace", 
                                    {"iteration": t + 1}
                                )
                
                # Report completion of this iteration
                if progress_callback:
                    await progress_callback()
        except Exception as e:
            self._logger.log_error(f"PACE optimization error: {str(e)}", "pace", {"features": str(features)})
        
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
        features: Dict[str, Any]
    ) -> str:
        """Run the actor to generate synthetic samples based on the current prompt."""
        try:
            completions = await self._llm.get_batch_completions(
                prompts=[prompt],
                features=features
            )
            return completions[0]
        
        except Exception as e:
            self._logger.log_error(
                f"Actor error: {str(e)}", 
                "pace", 
                {"prompt": prompt}
            )
            return "[ERROR: Actor failed to generate output]"
    
    async def _run_critic(
        self, 
        prompt: str,
        action: str, 
        features: Dict[str, Any]
    ) -> str:
        """Run the critic to provide a critique with suggestions for refining the prompt."""

        domain = features['domain']
        label = features['label']
        label_definition = features['label_definition']
        language = features['language']
        stakeholder = features['stakeholder']
        specification_format = features['specification_format']
        specification_level = features['specification_level']
        
        critique_prompt = f"""Instruction given:
"{prompt}"

It produced the following output:
{action}

Expected output format: a strictly valid JSON array of strings, e.g.:
[
  "First requirement text here",
  "Second requirement text here"
]

Each requirement must:
- Be {label} (Definition: {label_definition}).
- Be written in {language}.
- Pertain to a {domain} system.
- Be written from the perspective of {stakeholder}.
- Follow the {specification_format} format.
- Be specified at a {specification_level} level.

The full set should also be diverse enough for robust AI training.

Your task: Critique how the output fails to meet these expectations. Focus only on these points—don't suggest changes beyond them."""        
        
        critic_settings = {
            'llm': features['llm'],
            'temperature': 0.0,
            'top_p': 1.0
        }

        try:
            completions = await self._llm.get_batch_completions(
                prompts=[critique_prompt],
                features=critic_settings
            )
            return completions[0] if completions else "[ERROR: No critique generated]"
            
        except Exception as e:
            self._logger.log_error(f"Critic error: {str(e)}", "pace", {"prompt": prompt, "action": action})
            return "[ERROR: Critic failed to provide feedback]"
    
    async def _update_prompt(
        self, 
        current_prompt: str, 
        feedback_list: List[str],
        atomic_config: Dict[str, Any]
    ) -> str:
        """Update the prompt based on collected feedback."""
        
        combined_feedback = "\n\n".join([f"Feedback {i+1}:\n{fb}" for i, fb in enumerate(feedback_list)])
        
        update_prompt = f"""Instruction given:
"{current_prompt}"

Feedback:
{combined_feedback}

Rewrite the instruction to address the feedback.

Return only the new instruction as plain text — no extra text, quotes, or formatting."""

        update_settings = {
            'llm': atomic_config['llm'],
            'temperature': 0.0,
            'top_p': 1.0
        }

        try:
            completions = await self._llm.get_batch_completions(
                prompts=[update_prompt],
                features=update_settings
            )
            
            return completions[0] if completions else current_prompt
            
        except Exception as e:
            self._logger.log_error(f"Update error: {str(e)}", "pace", {"current_prompt": current_prompt})
            return current_prompt
    
    def _evaluate_prompt(
        self, 
        raw_completions: List[str], 
        samples_per_prompt: int,
    ) -> float:
        """Evaluate the prompt through the cosine distance of corresponding generated samples."""
        try:
            parsed_samples = []
            
            for raw_completion in raw_completions:
                parsed = parse_completion(raw_completion, samples_per_prompt)
                if parsed:
                    parsed_samples.extend(parsed)
                    
            embeddings = self._model.encode(parsed_samples)
            distances = cosine_distances(embeddings)
            avg_distance = np.mean(distances[np.triu_indices(len(parsed_samples), k=1)])
            return float(avg_distance) # Convert numpy.float32 to native Python float for JSON serialization
            
        except Exception as e:
            self._logger.log_error(
                f"Evaluation error: {str(e)}", 
                "pace"
            )
            return -1.0