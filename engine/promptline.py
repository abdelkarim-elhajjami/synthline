"""
Prompt template manager for data generation.
Builds prompts for LLM requests based on feature configurations.
"""
from typing import Dict, Any


class Promptline:
    """
    Builds parameterized prompts for data generation.
    
    Uses templates to create well-structured prompts customized by user-selected
    features such as domain, language, and specification format.
    """
    
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
    MULTI_TEMPLATE = '''Generate {count} diverse requirements that:
1. Are {label} (Definition: {label_definition}).
2. Are written in {language}.
3. Pertain to a {domain} system.
4. Are written from the perspective of {stakeholder}.
5. Follow the {specification_format} format.
6. Are specified at a {specification_level} level.

IMPORTANT: Format your completion exactly as a JSON array of strings, like this example:
[
  "1st requirement text",
  "2nd requirement text"
]

Do not use nested quotes within the JSON. Each requirement should be a simple string without additional formatting.
Include only the JSON array with requirements. No additional text or explanation.'''

    def __init__(self, logger=None):
        """
        Initialize the promptline manager.
        
        Args:
            logger: Optional logger for error reporting
        """
        self._logger = logger
    
    def build(self, features: Dict[str, Any], count: int = 1) -> str:
        """
        Build a prompt string using the provided features.
        
        Args:
            features: Dictionary of feature values to insert in the template
            count: Number of samples to request (determines template used)
            
        Returns:
            A formatted prompt string
            
        Raises:
            ValueError: If a required parameter is missing
        """
        template = self.MULTI_TEMPLATE if count > 1 else self.SINGLE_TEMPLATE
        
        # Required parameters - will raise KeyError if missing
        required_params = [
            "label", "label_definition", "language", "domain",
            "stakeholder", "specification_format", "specification_level"
        ]
        
        # Verify all required parameters are present
        missing = [param for param in required_params if param not in features]
        if missing:
            error_msg = f"Missing required parameters: {', '.join(missing)}"
            if self._logger:
                self._logger.log_error(error_msg, "promptline", features)
            raise ValueError(error_msg)
        
        # Create parameters dictionary with count
        params = {param: features[param] for param in required_params}
        params["count"] = count
        
        try:
            prompt = template.format(**params)
            return prompt
            
        except KeyError as e:
            error_msg = f"Missing required parameter: {e}"
            print(error_msg)

            if self._logger:
                self._logger.log_error(error_msg, "promptline", params)

            raise ValueError(f"Missing required parameter: {str(e).strip('')}")
            
        except Exception as e:
            error_msg = f"Error formatting prompt: {e}"
            print(error_msg)

            if self._logger:
                self._logger.log_error(error_msg, "promptline", params)
                
            raise