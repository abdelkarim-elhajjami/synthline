"""
Prompt manager for Synthline.
Builds prompts for LLM calls and manages PACE optimization.
"""
from typing import Dict, Any, Tuple
from utils.progress import ProgressCallback
from core.llm import LLMClient
from utils.logger import Logger
from core.pace import PACE

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

IMPORTANT: Generate only the requirement text without additional formatting. No additional text or explanation.'''
    
    # Template for generating multiple requirements
    MULTI_TEMPLATE = '''Generate {samples_per_prompt} diverse requirements that:
1. Are {label} (Definition: {label_definition}).
2. Are written in {language}.
3. Pertain to a {domain} system.
4. Are written from the perspective of {stakeholder}.
5. Follow the {specification_format} format.
6. Are specified at a {specification_level} level.

IMPORTANT: Format your completion exactly as a JSON array of strings, e.g.:
[
  "1st requirement text",
  "2nd requirement text"
]

Do not use nested quotes within the JSON. Each requirement should be a simple string without additional formatting.
Include only the JSON array with requirements. No additional text or explanation.'''

    def __init__(self, llm_client: LLMClient, logger: Logger):
        """Initialize the promptline manager."""
        self._llm = llm_client
        self._logger = logger
        self._optimizer = None
    
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
        connections = features['connections']
        
        # Run optimization
        optimized_prompt, score = await self._optimizer.optimize(
            features=features,
            progress_callback=progress_callback,
            initial_prompt=initial_prompt,
            n_iterations=n_iterations,
            n_actors=n_actors,
            connections=connections
        )
        
        return optimized_prompt, score