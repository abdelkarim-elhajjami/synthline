"""
Generator for creating synthetic data.
"""
from typing import Any, Dict, List, Optional, Tuple
from core.llm import LLMClient
from core.promptline import Promptline
from utils.logger import Logger
from utils.parsing import parse_completion
from utils.progress import ProgressCallback, track_progress

class Generator:
    """Generator for creating synthetic data using LLMs."""

    def __init__(
        self, 
        llm: LLMClient, 
        promptline: Promptline, 
        logger: Logger
    ) -> None:
        """Initialize the generator."""

        self._llm = llm
        self._promptline = promptline
        self._logger = logger
        self._fewer_samples_received = False

    async def generate(
        self,
        features: Dict[str, Any],
        progress_callback: ProgressCallback = None
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic data based on feature configuration.
        
        Args:
            features: Dictionary of feature settings
            progress_callback: Optional callback for reporting progress
            
        Returns:
            List of generated samples as dictionaries
        """
        all_samples = []
        self._fewer_samples_received = False
        
        total_samples = int(features['total_samples'])
        samples_per_prompt = int(features['samples_per_prompt'])
        
        # Generate samples until we have enough
        while len(all_samples) < total_samples:
            samples_needed = total_samples - len(all_samples)
            request_count = min(samples_needed, samples_per_prompt)
            
            # Generate samples
            new_samples, received_count = await self._generate_samples(
                features=features,
                samples_needed=samples_needed,
                samples_per_prompt=request_count
            )
            
            # Check if we received fewer samples than requested (token limit)
            if received_count < request_count and received_count > 0:
                self._fewer_samples_received = True
                
                self._logger.log_error(
                    f"Received fewer samples than requested ({received_count}/{request_count}), likely due to output token limit.",
                    "generator",
                    {"config": features}
                )
            
            # Add new samples to our collection
            if new_samples:
                all_samples.extend(new_samples)
                
                # Update progress if callback provided
                if progress_callback:
                    progress = min(100, (len(all_samples) / total_samples) * 100)
                    await track_progress(progress_callback, progress)
            else:
                break
            
        # Update progress to 100% when done
        if progress_callback:
            await track_progress(progress_callback, 100)
            
        return all_samples
    
    async def _generate_samples(
        self, 
        features: Dict[str, Any], 
        samples_needed: int, 
        samples_per_prompt: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Generate samples for a specific feature configuration.
        
        Args:
            features: Feature configuration dictionary
            samples_needed: Number of samples needed
            samples_per_prompt: Number of samples to request per prompt
            
        Returns:
            Tuple of (list of sample dictionaries, count of samples received)
        """
        new_samples = []
        
        # Check if we have an optimized prompt
        if 'optimized_prompt' in features:
            prompt = features['optimized_prompt']
        else:
            prompt = self._promptline.build(features)
        
        try:
            completion_list = await self._llm.get_batch_completions([prompt], features)
            completion = completion_list[0]
            
            sample_texts = parse_completion(completion, samples_per_prompt)
            
            for sample_text in sample_texts[:samples_needed]:
                if sample_text and sample_text.strip():
                    new_samples.append(self._create_sample(sample_text.strip(), features))
        
        except Exception as e:
            error_msg = f"Error generating from configuration: {e}"
            self._logger.log_error(error_msg, "generator", features)
        
        return new_samples, len(sample_texts)

    def _create_sample(self, sample_text: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured sample from text and configuration."""
        return {
            "text": sample_text,
            "label": features['label'],
            "domain": features['domain'],
            "language": features['language'],
            "stakeholder": features['stakeholder'],
            "specification_format": features['specification_format'],
            "specification_level": features['specification_level']
        }