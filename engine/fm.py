"""
Feature Manager for Synthline.
Defines and manages configurable features for data generation.
"""
from typing import Dict, List, Optional


class Feature:
    """
    Represents a configurable feature with its properties and sub-features.
    
    Attributes:
        name: The display name of the feature
        feature_type: The type of UI control ('select', 'input', 'group')
        options: Available options for select-type features
        subfeatures: Nested features organized in a dictionary
    """
    def __init__(
        self, 
        name: str, 
        feature_type: str, 
        options: Optional[List[str]] = None, 
        subfeatures: Optional[Dict[str, 'Feature']] = None
    ) -> None:
        self.name = name
        self.feature_type = feature_type
        self.options = options or []
        self.subfeatures = subfeatures or {}


class FM:
    """
    Feature Manager for Synthline.
    Manages the structure of configurable features.
    """
    def __init__(self) -> None:
        self.features: Dict[str, Feature] = {
            # Classification
            'ml_task': Feature('Classification', 'group', subfeatures={
                'label': Feature('Label', 'input'),
                'label_definition': Feature('Label Definition', 'input'),
            }),
            
            # Requirements Artifact
            'artifact': Feature('Requirements Artifact', 'group', subfeatures={
                'specification_format': Feature('Specification Format', 'select', [
                    'NL', 'Constrained NL', 'Use Case', 'User Story'
                ]),
                'specification_level': Feature('Specification Level', 'select', [
                    'High', 'Detailed'
                ]),
                'stakeholder': Feature('Stakeholder', 'select', [
                    'End Users', 'Business Managers', 'Developers', 'Regulatory Bodies'
                ]),
                'domain': Feature('Domain', 'input'),
                'language': Feature('Language', 'input'),
            }),
            
            # Generator
            'generator': Feature('Generator', 'group', subfeatures={
                'llm': Feature('LLM', 'select', ['deepseek-chat', 'gpt-4o']),
                'temperature': Feature('Temperature', 'input'),
                'top_p': Feature('Top P', 'input'),
                'samples_per_prompt': Feature('Samples Per Prompt', 'input'),
            }),
            
            # Output
            'output': Feature('Output', 'group', subfeatures={
                'output_format': Feature('Output Format', 'select', ['JSON', 'CSV']),
                'total_samples': Feature('Total Samples', 'input'),
            })
        }