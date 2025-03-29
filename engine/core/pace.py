"""
Implementation of PACE (Prompt Actor-Critic Editing) for Synthline.
https://aclanthology.org/2024.findings-acl.436/
"""
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from core.llm import LLMClient
from utils.logger import Logger
from utils.parsing import parse_completion
from utils.progress import ProgressCallback, track_progress

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
        
    async def optimize(
        self,
        features: Dict[str, Any],
        progress_callback: ProgressCallback = None,
        initial_prompt: Optional[str] = None,
        n_iterations: Optional[int] = None,
        n_actors: Optional[int] = None,
        connections: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float]:
        """Run the PACE optimization."""        
        # Initialization:
        current_prompt = initial_prompt
        best_prompt = current_prompt
        best_score = 0.0
        
        try:
            # Repeat until convergence or max iterations
            for t in range(n_iterations):                
                all_critiques = []
                all_actions = []
                
                # 1. Use n actors to generate outputs
                for _ in range(n_actors):
                    try:
                        action = await self._run_actor(prompt=current_prompt, features=features)
                        all_actions.append(action)
                        
                        # 2. Use n critics to provide critique
                        critique = await self._run_critic(prompt=current_prompt, action=action, features=features)
                        all_critiques.append(critique)
                        
                    except Exception as e:
                        self._logger.log_error(f"Actor-critic error: {str(e)}", "pace", {"prompt": current_prompt})
                
                try:
                    # 3. Update the prompt based on critiques
                    next_prompt = await self._update_prompt(current_prompt, all_critiques, features)
                    
                    # 3. Evaluate the new prompt
                    new_actions = []
                    for _ in range(n_actors):
                        action = await self._run_actor(prompt=next_prompt, features=features)
                        new_actions.append(action)
                    
                    next_score = self._evaluate_prompt(new_actions, features['samples_per_prompt'])
                    
                    if next_score > best_score:
                        best_prompt = next_prompt
                        best_score = next_score
                        
                        # Log when we find a better prompt
                        self._logger.log_prompt(
                            prompt=best_prompt,
                            score=best_score,
                            event="NEW BEST PROMPT"
                        )
                    
                    # Update current prompt for next iteration
                    current_prompt = next_prompt
                    
                    # Send update via WebSocket if connection exists
                    websocket = connections.get(features['connection_id'])
                    if websocket:
                        try:
                            await websocket.send_json({
                                "type": "prompt_update",
                                "prompt": best_prompt,
                                "score": best_score,
                                "iteration": t + 1,
                            })
                        except Exception as ws_error:
                            self._logger.log_error(
                                f"WebSocket send error: {str(ws_error)}", 
                                "pace", 
                                {"iteration": t + 1}
                            )
                    
                    # Report progress
                    if progress_callback:
                        await track_progress(progress_callback, ((t+1) / n_iterations) * 100)
                    
                except Exception as e:
                    self._logger.log_error(f"Prompt update error: {str(e)}", "pace", {"prompt": current_prompt})
        except Exception as e:
            self._logger.log_error(f"PACE optimization error: {str(e)}", "pace", {"features": str(features)})
        
        # Final progress update
        if progress_callback:
            await track_progress(progress_callback, 100)
        
        # Log final results
        self._logger.log_prompt(
            prompt=best_prompt,
            score=best_score,
            event="FINAL OPTIMIZED PROMPT"
        )
        
        return best_prompt, float(best_score)
    
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
        
        critique_prompt = f"""I gave you this instruction:

"{prompt}"

Based on this instruction, the following output was generated:
{action}

The output is expected to be a JSON array of strings, like this example:
[
  "1st requirement text",
  "2nd requirement text"
]

The generated requirements of the output are expected to be:
- Be {label} (Definition: {label_definition}).
- Be written in {language}.
- Pertain to a {domain} system.
- Be written from the perspective of {stakeholder}.
- Follow the {specification_format} format.
- Be specified at a {specification_level} level.
- Be diverse enough for robust AI model training.

Considering the generated requirements and the expected characteristics of the output, provide critical advice on how to improve the instruction.
IMPORTANT: Your task is to identify ONLY problems with how the output follows the instruction.
DO NOT suggest adding new requirements or formats not already in the instruction.
DO NOT suggest stylistic changes unless they directly relate to the expected characteristics."""
        
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
        features: Dict[str, Any]
    ) -> str:
        """Update the prompt based on collected feedback."""
        
        combined_feedback = "\n\n".join([f"Feedback {i+1}:\n{fb}" for i, fb in enumerate(feedback_list)])
        
        update_prompt = f"""You are tasked with improving an instruction for generating requirements.

Current instruction: "{current_prompt}"

Critical feedback received:
{combined_feedback}

Your task:
1. Create an improved version of the instruction that addresses the feedback
2. Return ONLY the improved instruction text
3. Do not include any explanations, quotes, prefixes, or formatting
4. Do not use JSON format or code blocks
5. The instruction should be ready to use as-is

Improved instruction:"""

        update_settings = {
            'llm': features['llm'],
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
            return float(avg_distance)
            
        except Exception as e:
            self._logger.log_error(
                f"Evaluation error: {str(e)}", 
                "pace"
            )
            return float(-1.0) 