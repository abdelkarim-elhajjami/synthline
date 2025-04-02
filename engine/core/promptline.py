"""
Prompt manager for Synthline.
Builds prompts for LLM calls and manages PACE optimization.
"""
from typing import Any, Dict, List, Tuple
from itertools import product
from core.llm import LLMClient
from core.pace import PACE
from utils.logger import Logger
from utils.progress import ProgressCallback

class Promptline:
    """Builds parameterized and optionally optimized prompts for data generation."""
    
    # Template for generating a single requirement
    SINGLE_TEMPLATE = '''Generate a requirement that:
1. Is {label} (Definition: {label_definition}).
2. Is written in {language}.
3. Pertains to a {domain} system.
4. Is written from the perspective of {stakeholder}.
5. Follows the {specification_format} format.
6. Is specified at a {specification_level} level.

Output only the requirement text. No additional text or formatting.'''
    
    # Template for generating multiple requirements
    MULTI_TEMPLATE = '''Generate {samples_per_prompt} diverse requirements that:
1. Are {label} (Definition: {label_definition}).
2. Are written in {language}.
3. Pertain to a {domain} system.
4. Are written from the perspective of {stakeholder}.
5. Follow the {specification_format} format.
6. Are specified at a {specification_level} level.

Format your completion exactly as a JSON array of strings, e.g.:
[
  "1st requirement text",
  "2nd requirement text"
]

Include only the JSON array. No additional text.'''

    def __init__(self, llm_client: LLMClient, logger: Logger):
        """Initialize the promptline manager."""
        self._llm = llm_client
        self._logger = logger
        self._optimizer = None
        self._multiple_select_features = [
            "specification_format",
            "specification_level",
            "stakeholder",
            "domain",
            "language",
        ]
    
    def get_atomic_configurations(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build a list of atomic configurations (one for each combination of multi-select features)."""
        configs_by_feature = {}
        
        for feature in self._multiple_select_features:
            value = features.get(feature)
            
            # Handle comma-separated text inputs for domain and language
            if feature in ['domain', 'language'] and isinstance(value, str):
                values = [item.strip() for item in value.split(',') if item.strip()]
                if values:
                    configs_by_feature[feature] = values
                    continue
            
            # Handle list values for multi-select features
            if isinstance(value, list) and value:
                configs_by_feature[feature] = value
            else:
                # Skip features that don't have multiple values
                configs_by_feature[feature] = [value] if value else [""]
        
        all_configurations = []
        feature_names = list(configs_by_feature.keys())
        
        product_of_values = product(*(configs_by_feature[f] for f in feature_names))
        
        for tuple_of_values in product_of_values:
            atomic_config = features.copy()
            for feature_name, selected_value in zip(feature_names, tuple_of_values):
                if selected_value:
                    atomic_config[feature_name] = selected_value
            all_configurations.append(atomic_config)
        
        return all_configurations
    
    def build(self, features: Dict[str, Any]) -> str:
        """Build a prompt string using the provided features."""
        
        template = self.MULTI_TEMPLATE if features['samples_per_prompt'] > 1 else self.SINGLE_TEMPLATE
        
        try:
            prompt = template.format(**features)
            return prompt
            
        except Exception as e:
            error_msg = f"Error formatting prompt: {e}"
            self._logger.log_error(error_msg, "promptline")
            raise
    
    def get_atomic_prompts(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate atomic prompts for all combinations of multi-select features."""
        atomic_configs = self.get_atomic_configurations(features)
        
        atomic_prompts = []
        for config in atomic_configs:
            prompt = self.build(config)
            atomic_prompts.append({
                "config": config,
                "prompt": prompt
            })
            
        return atomic_prompts
    
    async def optimize(self, 
                      features: Dict[str, Any], 
                      progress_callback: ProgressCallback = None) -> Tuple[str, float]:
        """
        Optimize a prompt using PACE if selected in features.
        
        Returns:
            Tuple of (optimized prompt, optimization score)
        """
        if self._optimizer is None:
            self._optimizer = PACE(
                llm_client=self._llm,
                logger=self._logger
            )
            
        # Generate initial prompt from templates
        initial_prompt = self.build(features)
        
        # Configure optimizer with features
        n_iterations = int(features['pace_iterations'])
        n_actors = int(features['pace_actors'])
        n_candidates = int(features['pace_candidates'])
        connections = features['connections']
        
        # Run optimization
        optimized_prompt, score = await self._optimizer.optimize(
            features=features,
            progress_callback=progress_callback,
            initial_prompt=initial_prompt,
            n_iterations=n_iterations,
            n_actors=n_actors,
            n_candidates=n_candidates,
            connections=connections
        )
        
        return optimized_prompt, score