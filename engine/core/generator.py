"""
Generator for creating synthetic data.
"""
from typing import Any, Dict, List, Tuple
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
        
        llm_settings = {k: features[k] for k in ['llm', 'temperature', 'top_p']}
        
        # Check for optimized atomic prompts
        if 'optimized_atomic_prompts' in features:
            # Use pre-optimized atomic prompts directly
            atomic_configs = []
            for optimized_prompt_data in features['optimized_atomic_prompts']:
                config = {**llm_settings, **optimized_prompt_data['config']}
                config['optimized_prompt'] = optimized_prompt_data['optimized_prompt']
                atomic_configs.append(config)
        else:
            # Get atomic configurations from promptline
            atomic_configs = self._promptline.get_atomic_configurations(features)
            
            for config in atomic_configs:
                for key, value in llm_settings.items():
                    config[key] = value
        
        n_configs = len(atomic_configs)
        sample_counts = self._distribute_samples(total_samples, n_configs)
        
        # Track progress across all configurations
        progress_total = 0
        
        # Generate samples for each atomic configuration
        for i, config in enumerate(atomic_configs):
            samples_for_config = sample_counts[i]
            
            if samples_for_config <= 0:
                continue
                
            # Generate samples until we have enough for this config
            config_samples = []
            while len(config_samples) < samples_for_config:
                samples_needed = samples_for_config - len(config_samples)
                request_count = min(samples_needed, samples_per_prompt)
                
                # Generate samples for this atomic configuration
                new_samples, received_count = await self._generate_samples(
                    atomic_config=config,
                    samples_needed=samples_needed,
                    samples_per_prompt=request_count
                )
                
                # Check if we received fewer samples than requested (token limit)
                if received_count < request_count and received_count > 0:
                    self._fewer_samples_received = True
                    
                    self._logger.log_error(
                        f"Received fewer samples than requested ({received_count}/{request_count}), likely due to output token limit.",
                        "generator",
                        {"config": config}
                    )
                
                # Add new samples to this config's collection
                if new_samples:
                    config_samples.extend(new_samples)
                else:
                    break
            
            # Add samples from this config to the overall collection
            all_samples.extend(config_samples)
            
            # Update progress if callback provided
            if progress_callback:
                progress_total += len(config_samples)
                progress = min(100, (progress_total / total_samples) * 100)
                await track_progress(progress_callback, progress)
            
        # Update progress to 100% when done
        if progress_callback:
            await track_progress(progress_callback, 100)
            
        return all_samples
    
    async def _generate_samples(
        self, 
        atomic_config: Dict[str, Any],
        samples_needed: int, 
        samples_per_prompt: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Generate samples for a specific atomic configuration.
        
        Args:
            atomic_config: Atomic configuration dictionary
            samples_needed: Number of samples needed
            samples_per_prompt: Number of samples to request per prompt
            
        Returns:
            Tuple of (list of sample dictionaries, count of samples received)
        """
        new_samples = []
        
        # Check if we have an optimized prompt
        if 'optimized_prompt' in atomic_config:
            prompt = atomic_config['optimized_prompt']
        else:
            prompt = self._promptline.build(atomic_config)
        
        try:
            completion_list = await self._llm.get_batch_completions([prompt], atomic_config)
            completion = completion_list[0]
            
            sample_texts = parse_completion(completion, samples_per_prompt)
            
            for sample_text in sample_texts[:samples_needed]:
                if sample_text and sample_text.strip():
                    new_samples.append(self._create_sample(sample_text.strip(), atomic_config))
        
        except Exception as e:
            error_msg = f"Error generating from configuration: {e}"
            self._logger.log_error(error_msg, "generator", atomic_config)
        
        return new_samples, len(sample_texts)

    def _create_sample(self, sample_text: str, atomic_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured sample from text and configuration."""
        return {
            "text": sample_text,
            "label": atomic_config['label'],
            "domain": atomic_config['domain'],
            "language": atomic_config['language'],
            "stakeholder": atomic_config['stakeholder'],
            "specification_format": atomic_config['specification_format'],
            "specification_level": atomic_config['specification_level']
        }

    def _distribute_samples(self, total_samples: int, n_configs: int) -> List[int]:
        """Distribute total samples as evenly as possible among configurations."""
        if n_configs == 0:
            return []
        
        base_count = total_samples // n_configs
        remainder = total_samples % n_configs
        
        return [base_count + (1 if i < remainder else 0) for i in range(n_configs)]
        