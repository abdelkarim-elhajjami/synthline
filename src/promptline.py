from typing import Dict

class Promptline:
    """Handles the generation of prompts for data generation."""

    def __init__(self):
        self.template = '''Generate a requirement that:
            1. Is classified as {label} (Description: {label_description})
            2. Is of type {requirement_type}
            3. Is written in {language}
            4. Is for a {domain} system
            5. Is from {requirement_source} perspective
            6. Follows {specification_format} format
            7. Is written at {specification_level} level
            Important: Generate ONLY the requirement text. No additional context.
            '''

    def build(self, features: Dict[str, str]) -> str:
        """
        Build a prompt string using the provided features.
        """
        return self.template.format(
            label=features["label"],
            label_description=features["label_description"],
            requirement_type=features["requirement_type"],
            language=features["language"],
            domain=features["domain"],
            requirement_source=features["requirement_source"],
            specification_format=features["specification_format"],
            specification_level=features["specification_level"]
        )
