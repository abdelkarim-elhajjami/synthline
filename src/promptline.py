from typing import Dict, Any

class Promptline:
    """Handles the generation of prompts for data generation."""
    
    SINGLE_TEMPLATE = '''Generate a requirement that:
1. Is classified as {label} (Description: {label_description})
2. Is written in {language}
3. Is for a {domain} system
4. Is from {requirement_source} perspective
5. Follows {specification_format} format
6. Is written at {specification_level} level

IMPORTANT: Generate only the requirement text without additional formatting. No additional text or explanation.'''
            
    MULTI_TEMPLATE = '''Generate {count} different requirements that:
1. Are classified as {label} (Description: {label_description})
2. Are written in {language}
3. Are for a {domain} system
4. Are from {requirement_source} perspective
5. Follow {specification_format} format
6. Are written at {specification_level} level

IMPORTANT: Format your completion exactly as a JSON array of strings, like this example:
[
  "1st requirement text",
  "2nd requirement text"
]

Do not use nested quotes within the JSON. Each requirement should be a simple string without additional formatting.
Include only the JSON array with requirements. No additional text or explanation.'''

    def __init__(self, logger=None):
        self._logger = logger
    
    def build(self, features: Dict[str, Any], count: int = 1) -> str:
        """Build a prompt string using the provided features."""
        template = self.MULTI_TEMPLATE if count > 1 else self.SINGLE_TEMPLATE
        
        params = {
            "count": count,
            "label": features["label"],
            "label_description": features["label_description"],
            "language": features["language"],
            "domain": features["domain"],
            "requirement_source": features["requirement_source"],
            "specification_format": features["specification_format"],
            "specification_level": features["specification_level"]
        }
        
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