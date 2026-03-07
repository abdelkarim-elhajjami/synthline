from typing import Any, Dict, List, Tuple
from synthline.core.llm import LLMClient
from synthline.core.promptline import Promptline
from synthline.utils.logger import Logger
from synthline.utils.parsing import parse_completion_with_meta
from synthline.utils.progress import ProgressCallback, track_progress

class Generator:
    def __init__(
        self, 
        llm: LLMClient, 
        promptline: Promptline, 
        logger: Logger
    ) -> None:
        self._llm = llm
        self._promptline = promptline
        self._logger = logger
        self._fewer_samples_received = False
        self._parsing_degraded = False

    @property
    def fewer_samples_received(self) -> bool:
        return self._fewer_samples_received
    
    @property
    def parsing_degraded(self) -> bool:
        return self._parsing_degraded

    async def generate(
        self,
        features: Dict[str, Any],
        progress_callback: ProgressCallback = None,
        api_keys: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        all_samples = []
        self._fewer_samples_received = False
        self._parsing_degraded = False
        
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
                if 'pace_score' in optimized_prompt_data:
                    config['pace_score'] = optimized_prompt_data['pace_score']
                atomic_configs.append(config)
        else:
            # Get atomic configurations from promptline
            atomic_configs = self._promptline.get_atomic_configurations(features)
            
            for config in atomic_configs:
                for key, value in llm_settings.items():
                    config[key] = value
        
        n_configs = len(atomic_configs)
        sample_counts = self._distribute_samples(total_samples, n_configs)
        
        # Track progress across all configurations.
        generated_so_far = 0
        
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
                new_samples, received_count, parsing_degraded = await self._generate_samples(
                    atomic_config=config,
                    samples_needed=samples_needed,
                    samples_per_prompt=request_count,
                    spp_for_prompt=samples_per_prompt,
                    api_keys=api_keys
                )
                if parsing_degraded:
                    self._parsing_degraded = True
                    if request_count > 1:
                        # Degraded parsing for a multi-sample request means we could
                        # not reliably recover all requested structured samples.
                        self._fewer_samples_received = True
                    self._logger.log_error(
                        "LLM output parsing degraded to plain-text fallback; structured sample parsing failed.",
                        "generator",
                        {"config": config},
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
                    generated_so_far += len(new_samples)
                    if progress_callback and total_samples > 0:
                        progress = min(100.0, (generated_so_far / total_samples) * 100.0)
                        await track_progress(progress_callback, progress)
                else:
                    break
            
            # Add samples from this config to the overall collection
            all_samples.extend(config_samples)
            
        # Update progress to 100% when done
        if progress_callback:
            await track_progress(progress_callback, 100)
            
        return all_samples
    
    async def _generate_samples(
        self, 
        atomic_config: Dict[str, Any],
        samples_needed: int, 
        samples_per_prompt: int,
        spp_for_prompt: int = 1,
        api_keys: Dict[str, str] = None
    ) -> Tuple[List[Dict[str, Any]], int, bool]:
        new_samples = []
        parsing_degraded = False
        
        # Check if we have an optimized prompt
        if 'optimized_prompt' in atomic_config:
            prompt = atomic_config['optimized_prompt']
        else:
            prompt = self._promptline.build(atomic_config, samples_per_prompt=spp_for_prompt)
        
        sample_texts = []
        try:
            completion_list = await self._llm.get_batch_completions(
                prompts=[prompt], 
                features=atomic_config,
                api_keys=api_keys
            )
            completion = completion_list[0]
            
            sample_texts, parsing_degraded = parse_completion_with_meta(completion, samples_per_prompt)
            
            for sample_text in sample_texts[:samples_needed]:
                if sample_text and sample_text.strip():
                    new_samples.append({
                        "text": sample_text.strip(),
                        "config": {**atomic_config, "prompt": prompt},
                    })
        
        except Exception as e:
            error_msg = f"Error generating from configuration: {e}"
            self._logger.log_error(
                error_msg,
                "generator",
                {"atomic_config": atomic_config},
            )
            raise
        
        return new_samples, len(sample_texts), parsing_degraded

    def _distribute_samples(self, total_samples: int, n_configs: int) -> List[int]:
        """Distribute total samples as evenly as possible among configurations."""
        if n_configs == 0:
            return []
        
        base_count = total_samples // n_configs
        remainder = total_samples % n_configs
        
        return [base_count + (1 if i < remainder else 0) for i in range(n_configs)]
        