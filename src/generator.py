from typing import Dict, Any, List, Optional, Callable, Tuple
import json
import re
from itertools import product
from llm_client import LLMClient
from promptline import Promptline

class Generator:
    """Generator class for generating synthetic data samples based on provided configurations.
    
    This class handles the generation of synthetic data by:
    1. Flattening multi-select feature configurations into atomic configurations
    2. Distributing sample generation evenly across configurations
    3. Handling LLM API calls and parsing completions
    4. Tracking progress and handling errors transparently
    """
    def __init__(
        self, 
        llm: LLMClient, 
        promptline: Promptline, 
        batch_size: int,
        logger=None
    ) -> None:
        self._llm = llm
        self._promptline = promptline
        self._batch_size = batch_size
        self._logger = logger
        
        # Features that can have multiple selections
        self._multiple_select_features = [
            "specification_format",
            "specification_level",
            "requirement_source",
            "domain",
            "language",
        ]
        
        self._fewer_samples_received = False

    def _flatten_configurations(self, feature_values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert multi-select feature configurations into a list of atomic configurations.
        
        Args:
            feature_values: Dictionary of feature values, where some may be lists
            
        Returns:
            List of dictionaries, each representing a single atomic configuration
        """
        configs_by_feature = {
            feature: [value] if not isinstance(value, list) else value
            for feature, value in feature_values.items()
            if feature in self._multiple_select_features
        }

        feature_names = list(configs_by_feature.keys())
        all_configurations = []
        
        # Generate all combinations of feature values
        for values in product(*(configs_by_feature[f] for f in feature_names)):
            atomic_config = feature_values.copy()
            for feature_name, value in zip(feature_names, values):
                if value is not None:
                    atomic_config[feature_name] = value
            all_configurations.append(atomic_config)
        
        return all_configurations

    def generate_samples(
        self, 
        feature_values: Dict[str, Any], 
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """Generate synthetic data samples based on provided configurations.
        
        Args:
            feature_values: Dictionary of feature configuration values
            progress_callback: Optional callback function to report progress
            
        Returns:
            List of generated samples
        """
        configurations = self._flatten_configurations(feature_values)
        
        if not configurations:
            raise ValueError("No valid configurations found")
        
        all_samples = []
        self._fewer_samples_received = False
        
        subset_size = int(feature_values.get('subset_size'))
        samples_per_call = int(feature_values.get('samples_per_call'))
        
        # Keep trying until we reach the target or can't make progress
        while len(all_samples) < subset_size:
            # Try to generate a batch of samples
            made_progress = self._try_generate_batch(
                all_samples=all_samples,
                configurations=configurations,
                samples_per_call=samples_per_call,
                subset_size=subset_size,
                progress_callback=progress_callback
            )
            
            # If we couldn't make progress, break out of the loop
            if not made_progress:
                break
            
        # Update progress to 100% when done
        if progress_callback:
            progress_callback(100)
            
        return all_samples

    def _try_generate_batch(
        self, 
        all_samples: List[Dict[str, Any]],
        configurations: List[Dict[str, Any]],
        samples_per_call: int,
        subset_size: int,
        progress_callback: Optional[Callable[[float], None]]
    ) -> bool:
        """Try to generate a batch of samples across configurations. Return True if progress was made."""
        made_progress = False
        total_configs = len(configurations)
        
        for i, config in enumerate(configurations):
            # Skip if we have enough samples already
            if len(all_samples) >= subset_size:
                break
            
            # Calculate how many samples we already have for this configuration
            existing_count = self._count_samples_for_config(all_samples, config)
            
            # Calculate target samples per configuration (distribute evenly)
            target_per_config = subset_size // total_configs
            if i < subset_size % total_configs:
                target_per_config += 1
            
            # Calculate how many more samples we need for this configuration
            samples_needed = max(0, target_per_config - existing_count)
            
            # Skip if we don't need more samples for this configuration
            if samples_needed == 0:
                continue
            
            # Limit request to what we actually need
            request_count = min(samples_needed, samples_per_call)
            
            # Generate samples for this configuration
            new_samples, received_count = self._generate_for_config(
                config=config,
                samples_needed=samples_needed,
                samples_per_call=request_count
            )
            
            # Check if we received fewer samples than requested (token limit)
            if received_count < request_count and received_count > 0:
                self._fewer_samples_received = True
                
                # Log this issue
                if self._logger:
                    self._logger.log_error(
                        f"Received fewer samples than requested ({received_count}/{request_count}), likely due to output token limit.",
                        "generator",
                        {"config": config}
                    )
            
            # Add new samples to our collection
            if new_samples:
                all_samples.extend(new_samples)
                made_progress = True
            
            # Update progress if callback provided
            if progress_callback:
                progress = min(100, (len(all_samples) / subset_size) * 100)
                progress_callback(progress)
        
        return made_progress

    def _count_samples_for_config(self, samples: List[Dict[str, Any]], config: Dict[str, Any]) -> int:
        """Count how many samples match a specific configuration.
        
        Args:
            samples: List of existing samples
            config: Configuration to match against
            
        Returns:
            Count of samples matching the configuration
        """
        return sum(
            1 for sample in samples 
            if all(sample.get(feature) == config.get(feature) 
                   for feature in self._multiple_select_features)
        )
    
    def _generate_for_config(
        self, 
        config: Dict[str, Any], 
        samples_needed: int, 
        samples_per_call: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Generate samples for a specific configuration.
        
        Args:
            config: Configuration to use for generation
            samples_needed: Number of samples needed
            samples_per_call: Maximum samples to request per API call
            
        Returns:
            Tuple of (generated samples, number of samples received)
        """
        new_samples = []
        
        prompt = self._promptline.build(config, samples_per_call)
        try:
            completion_list = self._llm.generate([prompt], config)
            generation = completion_list[0] if completion_list else ""
            
            if not generation.strip():
                print(f"Warning: Empty completion received for configuration: {config}")
                return [], 0
                
            # Parse the completion to extract requirements
            sample_texts = self._parse_json_samples(generation, samples_per_call)
            
            # Create samples from the extracted texts
            for sample_text in sample_texts[:samples_needed]:
                if sample_text and sample_text.strip():
                    new_samples.append(self._create_sample(sample_text.strip(), config))
        
        except Exception as e:
            error_msg = f"Error generating from configuration {config}: {e}"
            print(error_msg)
            
            if self._logger:
                self._logger.log_error(error_msg, "generator", config)
        
        return new_samples, len(sample_texts) if 'sample_texts' in locals() else 0

    def _create_sample(self, sample_text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        sample = {
            "text": sample_text,
            "label": config.get("label"),
            "domain": config.get("domain"),
            "language": config.get("language"),
            "requirement_source": config.get("requirement_source"),
            "specification_format": config.get("specification_format"),
            "specification_level": config.get("specification_level")
        }
        return sample
    
    def _parse_json_samples(self, text: str, expected_count: int) -> List[str]:
        """Parse samples from a JSON-formatted completion."""
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
        """Try to extract a JSON array from text."""
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
        
        # Try with common fixes
        try:
            cleaned_text = json_text.replace('\\"', '"').replace('""', '"')
            data = json.loads(cleaned_text)
            if isinstance(data, list):
                return [item.strip() for item in data if isinstance(item, str) and item.strip()]
        except:
            pass
        
        return []
    
    def _extract_samples_from_lines(self, text: str) -> List[str]:
        """Extract samples by splitting text into lines."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        samples = []
        
        for line in lines:
            if not line.startswith(('[', ']', '{', '}')) and len(line) > 10:
                cleaned = re.sub(r'^\d+\.|\-\s+|^"|"$|,$', '', line).strip()
                if cleaned:
                    samples.append(cleaned)
        
        return samples if samples else [text.strip()]