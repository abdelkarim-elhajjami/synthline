"""
Implementation of PACE (Prompt Actor-Critic Editing) for Synthline.
https://aclanthology.org/2024.findings-acl.436/
"""
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from llm_client import LLMClient

# Type for progress callback
ProgressCallback = Optional[Callable[[float], Awaitable[None]]]

class PACE:
    """Implements the PACE approach for prompt optimization."""
    def __init__(
        self,
        llm_client: LLMClient,
        iterations: int,
        num_actors: int,
        initial_prompt: str,
        logger=None,
        connections: Dict[str, Any] = None
    ) -> None:
        """
        Initialize the PACE optimizer.
        
        Args:
            llm_client: LLM client for text generation
            iterations: Maximum number of iterations for optimization
            num_actors: Number of actor-critic pairs per iteration
            initial_prompt: Starting prompt
            logger: Optional logger for logging prompts, conversations, and error reporting
            connections: Dictionary of active WebSocket connections
        """
        self._llm = llm_client
        self._max_iterations = iterations
        self._num_actors = num_actors
        self._logger = logger
        self._best_prompt = initial_prompt
        self._best_score = 0.0
        self._connections = connections or {}
        
    async def optimize(
        self,
        feature_values: Dict[str, Any],
        progress_callback: ProgressCallback = None
    ) -> Tuple[str, float]:
        """
        Run the PACE optimization.
        
        Args:
            feature_values: Dictionary of feature values
            progress_callback: Optional callback for reporting progress
            
        Returns:
            Tuple of (optimized prompt, best score)
        """
        # Initialization:
        # p_0 = initial prompt
        current_prompt = self._best_prompt
        # p* = p_0 (best prompt starts as initial prompt)
        best_prompt = current_prompt
        best_score = 0.0  # Initial score is 0
        
        # Calculate total expected steps for progress tracking
        total_steps = self._max_iterations * self._num_actors
        completed_steps = 0
        
        try:
            # Repeat until convergence or max iterations
            for t in range(self._max_iterations):
                # Track current iteration for logging
                feature_values['current_iteration'] = t + 1
                
                all_critiques = []
                all_actions = []  # Track actions for logging
                
                # 1. Use n actors to generate outputs
                for _ in range(self._num_actors):
                    try:
                        # a_i^(t) = actor(p_t)
                        action = await self._run_actor(
                            prompt=current_prompt,
                            feature_values=feature_values
                        )
                        all_actions.append(action)
                        
                        # 2. Use n critics to evaluate and produce feedback
                        # c_i^(t) = critic(p_t, a_i^(t))
                        critique = await self._run_critic(
                            prompt=current_prompt,
                            action=action,
                            feature_values=feature_values
                        )
                        
                        all_critiques.append(critique)
                        
                    except Exception as e:
                        if self._logger:
                            self._logger.log_error(
                                f"Actor-critic error: {str(e)}", 
                                "pace", 
                                {"prompt": current_prompt}
                            )
                    
                    # Update progress
                    completed_steps += 1
                    if progress_callback:
                        progress = min(100, (completed_steps / total_steps) * 100)
                        await self._call_progress(progress_callback, progress)
                
                try:
                    # 3. Update the prompt using all n critiques
                    # p_(t+1) = update(p_t, {c^(i)_t}i=1^n)
                    next_prompt = await self._update_prompt(
                        current_prompt, 
                        all_critiques, 
                        feature_values
                    )
                    
                    # 4. Evaluate the new prompt
                    # Generate output for the new prompt: a_(t+1) = actor(p_(t+1))
                    new_actions = []
                    for i in range(self._num_actors):
                        action = await self._run_actor(
                            prompt=next_prompt,
                            feature_values=feature_values
                        )
                        new_actions.append(action)
                    
                    # Score the output: s_(t+1) = s(a_(t+1))
                    next_score = self._evaluate_prompt(
                        new_actions, 
                        feature_values.get('samples_per_prompt', 1)
                    )
                    
                    # Log the prompt evaluation
                    if self._logger:
                        self._logger.log_prompt(
                            prompt=next_prompt,
                            feedback=None,
                            score=next_score,
                            iteration=t+1
                        )
                    
                    # 5. Compare to the best so far
                    # If s_(t+1) > s(p*), then p* = p_(t+1)
                    if next_score > best_score:
                        best_prompt = next_prompt
                        best_score = next_score
                        
                        # Log when we find a better prompt
                        if self._logger:
                            self._logger.log_prompt(
                                prompt=best_prompt,
                                score=best_score,
                                iteration=t+1,
                                feedback="NEW BEST PROMPT"
                            )
                    
                    # Update current prompt for next iteration
                    current_prompt = next_prompt
                    
                    # Send update via WebSocket if connection exists
                    websocket = self._connections.get(feature_values.get('connection_id'))
                    if websocket:
                        try:
                            await websocket.send_json({
                                "type": "prompt_update",
                                "prompt": self._clean_prompt(best_prompt),
                                "score": best_score,
                                "iteration": t + 1,
                            })
                        except Exception as ws_error:
                            if self._logger:
                                self._logger.log_error(
                                    f"WebSocket send error: {str(ws_error)}", 
                                    "pace", 
                                    {"iteration": t + 1}
                                )
                    
                    # Report progress
                    if progress_callback:
                        await self._call_progress(progress_callback, ((t+1) / self._max_iterations) * 100)
                    
                except Exception as e:
                    print(f"Error in prompt update/evaluation, iteration {t+1}: {e}")
                    if self._logger:
                        self._logger.log_error(
                            f"Prompt update error: {str(e)}", 
                            "pace", 
                            {"prompt": current_prompt}
                        )
        except Exception as e:
            print(f"Error in PACE optimization: {e}")
            if self._logger:
                self._logger.log_error(
                    f"PACE optimization error: {str(e)}", 
                    "pace", 
                    {"feature_values": str(feature_values)}
                )
        
        # Final progress update
        if progress_callback:
            await self._call_progress(progress_callback, 100)
        
        # Store the best prompt and score for the instance
        self._best_prompt = best_prompt
        self._best_score = best_score
        
        # Log final results
        if self._logger:
            self._logger.log_prompt(
                prompt=best_prompt,
                score=best_score,
                iteration=self._max_iterations,
                feedback="FINAL OPTIMIZED PROMPT"
            )
        
        return best_prompt, float(best_score)
    
    async def _call_progress(self, callback: Callable, progress: float) -> None:
        """Safely call the progress callback."""
        if asyncio.iscoroutinefunction(callback):
            await callback(progress)
        else:
            await asyncio.to_thread(callback, progress)
    
    async def _run_actor(
        self, 
        prompt: str, 
        feature_values: Dict[str, Any]
    ) -> str:
        """Run the actor to generate synthetic samples based on the current prompt."""
        try:
            completions = await self._llm.generate(
                prompts=[prompt],
                features=feature_values
            )
            # Just return the raw completion for critic to evaluate formatting
            return completions[0] if completions else ""
        
        except Exception as e:
            if self._logger:
                self._logger.log_error(
                    f"Actor error: {str(e)}", 
                    "pace", 
                    {"prompt": prompt}
                )
            return ""

    def _parse_json_samples(self, text: str, expected_count: int) -> List[str]:
        """
        Parse samples from LLM completion text.
        
        Uses multiple strategies to parse JSON arrays or plain text
        based on the expected format.
        
        Args:
            text: LLM completion text
            expected_count: Expected number of samples
            
        Returns:
            List of sample texts
        """
        # First try: Extract and parse JSON array
        samples = self._try_parse_json_array(text)
        if samples:
            return samples
        
        # Second try: Handle single sample case
        if expected_count == 1:
            return [text.strip()]
        
        # Fallback: Extract line by line
        return self._extract_samples_from_lines(text)

    def _try_parse_json_array(self, text: str) -> List[str]:
        """
        Try to extract a JSON array from text.
        
        Args:
            text: Text that may contain a JSON array
            
        Returns:
            List of strings from the array or empty list if parsing fails
        """
        import json
        
        json_start = text.find('[')
        json_end = text.rfind(']')
        
        if json_start < 0 or json_end <= json_start:
            return []
        
        json_text = text[json_start:json_end+1]
        
        # Try standard JSON parsing
        try:
            data = json.loads(json_text)
            if isinstance(data, list):
                return [item.strip() for item in data if isinstance(item, str) and item.strip()]
        except json.JSONDecodeError:
            pass
        
        # Try with common fixes for JSON formatting issues
        try:
            cleaned_text = json_text.replace('\\"', '"').replace('""', '"')
            data = json.loads(cleaned_text)
            if isinstance(data, list):
                return [item.strip() for item in data if isinstance(item, str) and item.strip()]
        except:
            pass
        
        return []

    def _extract_samples_from_lines(self, text: str) -> List[str]:
        """
        Extract samples by splitting text into lines and cleaning them.
        
        Args:
            text: Multi-line text to parse
            
        Returns:
            List of extracted sample texts
        """
        import re
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        samples = []
        
        for line in lines:
            # Skip JSON syntax and look for substantial content
            if not line.startswith(('[', ']', '{', '}')) and len(line) > 10:
                # Remove numbering, quotation marks, and other formatting
                cleaned = re.sub(r'^\d+\.|\-\s+|^"|"$|,$', '', line).strip()
                if cleaned:
                    samples.append(cleaned)
        
        # If we couldn't extract anything, return the whole text as one sample
        return samples if samples else [text.strip()]
    
    async def _run_critic(
        self, 
        prompt: str,
        action: str, 
        feature_values: Dict[str, Any]
    ) -> str:
        """Run the critic to provide a critique with suggestions for refining the prompt."""

        domain = feature_values.get('domain')
        label = feature_values.get('label')
        label_description = feature_values.get('label_description')
        stakeholder = feature_values.get('stakeholder')
        specification_format = feature_values.get('specification_format')
        specification_level = feature_values.get('specification_level')
        
        critique_prompt = f"""You are a prompt engineering expert.

Review the following prompt:
---
{prompt}
---

and the corresponding model output:
---
{action}
---

Provide critical feedback to refine the prompt so that it:
1. Precisely adheres to the required attributes:
    - Domain: {domain}
    - Classification Label: {label} (Description: {label_description})
    - Stakeholder Perspective: {stakeholder}
    - Specification Format: {specification_format}
    - Specification Level: {specification_level}
2. Generates high-quality diverse synthetic data suitable for robust AI model training.
3. Maintains the expected output format as specified in the prompt."""
        
        critic_settings = {
            'llm': feature_values.get('llm'),
            'temperature': 0.0,
            'top_p': 1.0
        }

        try:
            completions = await self._llm.generate(
                prompts=[critique_prompt],
                features=critic_settings
            )
            critique = completions[0] if completions else ""
            
            if self._logger:
                iteration = feature_values.get('current_iteration', 0)
                self._logger.log_prompt(
                    prompt=prompt,
                    feedback=critique,
                    score=None,
                    iteration=iteration
                )
            
            return critique
            
        except Exception as e:
            if self._logger:
                self._logger.log_error(
                    f"Critic error: {str(e)}", 
                    "pace", 
                    {"prompt": prompt, "action": action}
                )
            return ""
    
    async def _update_prompt(
        self, 
        current_prompt: str, 
        feedback_list: List[str],
        feature_values: Dict[str, Any]
    ) -> str:
        """Update the prompt based on collected feedback."""
        
        combined_feedback = "\n\n".join([f"Feedback {i+1}:\n{fb}" for i, fb in enumerate(feedback_list)])
        
        update_prompt = f"""You are a prompt engineering expert. Improve the following prompt using the provided feedback.

Prompt:
---
{current_prompt}
---

Feedback:
---
{combined_feedback}
---

Return only the improved prompt with the same format as the original prompt without any additional text or formatting."""

        update_settings = {
            'llm': feature_values.get('llm'),
            'temperature': 0.0,
            'top_p': 1.0
        }

        try:
            completions = await self._llm.generate(
                prompts=[update_prompt],
                features=update_settings
            )
            
            raw_completion = completions[0] if completions else current_prompt
            clean_prompt = self._clean_prompt(raw_completion)
            
            if self._logger:
                self._logger.log_prompt(
                    prompt=current_prompt,
                    updated_prompt=clean_prompt,
                    feedback=combined_feedback,
                    score=None,
                    iteration=feature_values.get('current_iteration', 0)
                )
            
            return clean_prompt
            
        except Exception as e:
            if self._logger:
                self._logger.log_error(
                    f"Update error: {str(e)}", 
                    "pace", 
                    {"current_prompt": current_prompt}
                )
            return current_prompt
    
    def _evaluate_prompt(
        self, 
        raw_completions: List[str], 
        samples_per_prompt: int
    ) -> float:
        """
        Evaluate the prompt through the diversity of the generated samples.
        
        Args:
            raw_completions: List of raw outputs from actor
            samples_per_prompt: Expected number of samples per prompt
            
        Returns:
            A normalized score between 0 and 1
        """
        try:
            if not raw_completions or len(raw_completions) < 2:
                return 0.0
            
            # Parse the completions to get the actual samples
            parsed_samples = []
            
            for raw_completion in raw_completions:
                parsed = self._parse_json_samples(raw_completion, samples_per_prompt)
                if parsed:
                    parsed_samples.extend(parsed)
            
            if len(parsed_samples) < 2:
                if self._logger:
                    self._logger.log_error(
                        "Insufficient parsed samples for diversity evaluation", 
                        "pace"
                    )
                return 0.0
            
            # 1. Calculate pairwise similarities using the parsed samples
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load models only when needed (could be optimized to load once in the class)
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
            model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens').to(device)
            
            # Get embeddings for parsed samples
            embeddings = []
            batch_size = 8
            
            for i in range(0, len(parsed_samples), batch_size):
                batch = tokenizer(
                    parsed_samples[i:i+batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**batch).last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(outputs)
            
            embeddings = np.vstack(embeddings)
            similarity_matrix = cosine_similarity(embeddings)
            
            # Calculate average pairwise similarity (excluding self-similarity)
            similarity_mask = ~np.eye(len(similarity_matrix), dtype=bool)
            avg_pairwise_similarity = similarity_matrix[similarity_mask].mean()
            
            # 2. Calculate inter-ngram frequency
            def _generate_ngrams(text, n=3):
                """Generate n-grams from text."""
                cleaned = text.replace('.', '').replace('\n', ' ').strip()
                words = [w for w in cleaned.split() if w]
                return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            
            # Calculate ngram diversity using parsed samples
            all_ngrams = []
            for text in parsed_samples:
                all_ngrams.extend(_generate_ngrams(text))
            
            if not all_ngrams:
                inter_ngram_freq = 1.0  # Worst case
            else:
                unique_ngrams = set(all_ngrams)
                inter_ngram_freq = sum(all_ngrams.count(ngram) for ngram in unique_ngrams) / len(unique_ngrams)
            
            # Normalize metrics into [0,1] range for the final score
            # Lower similarity and lower ngram frequency are better
            
            # Normalize similarity (typically in range -1 to 1)
            # Transform so that 1 (identical) becomes 0 (bad) and -1 (opposite) becomes 1 (good)
            normalized_similarity_score = 1 - ((avg_pairwise_similarity + 1) / 2)
            
            # Normalize inter-ngram frequency (assuming typical range of 1-5)
            # Clip to prevent extreme values
            clipped_inter_ngram_freq = min(5, max(1, inter_ngram_freq))
            normalized_ngram_score = 1 - ((clipped_inter_ngram_freq - 1) / 4)
            
            # Combine scores with weights
            similarity_weight = 0.6
            ngram_weight = 0.4
            
            final_score = (similarity_weight * normalized_similarity_score + 
                          ngram_weight * normalized_ngram_score)
            
            return float(final_score)
            
        except Exception as e:
            if self._logger:
                self._logger.log_error(
                    f"Evaluation error: {str(e)}", 
                    "pace"
                )
            return float(0.0)

    def _clean_prompt(self, prompt: str) -> str:
        """
        Clean a prompt to remove outermost wrappers while preserving content.
        
        Args:
            prompt: The prompt to clean
            
        Returns:
            The cleaned prompt text
        """
        import json
        import re
        
        # If it's already a clean string, return it
        prompt = prompt.strip()
        
        # Case 1: Handle the specific JSON format with "prompt" key
        if prompt.startswith('{') and prompt.endswith('}'):
            try:
                data = json.loads(prompt)
                if isinstance(data, dict) and 'prompt' in data:
                    return data['prompt']
            except:
                # If parsing fails, don't modify the prompt
                pass
        
        # Case 2: If it starts with "```" and ends with "```", extract content only if
        # these markers wrap the entire content
        if prompt.startswith('```') and prompt.endswith('```'):
            pattern = r'^```(?:\w+)?\s*([\s\S]*?)```\s*$'
            match = re.match(pattern, prompt)
            if match:
                return match.group(1).strip()
        
        return prompt