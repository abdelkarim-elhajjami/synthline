"""
Feature Manager for Synthline.
Defines and manages configurable features for data generation.
"""
from typing import Dict, List
from pydantic import BaseModel

class Feature(BaseModel):
    """
    Configurable feature with properties and sub-features.
    
    Attributes:
        name: Display name
        feature_type: UI control type ('select', 'input', 'group', 'multi-select')
        options: Available options for select-type features
        multiple: Whether multiple selections are allowed
        subfeatures: Nested features in a dictionary
    """
    name: str
    feature_type: str
    options: List[str] = []
    multiple: bool = False
    subfeatures: Dict[str, 'Feature'] = {}


class FM:
    """Feature Manager for managing configurable features."""
    def __init__(self) -> None:
        self.features: Dict[str, Feature] = {
            # Classification
            'ml_task': Feature(
                name='Classification', 
                feature_type='group', 
                subfeatures={
                    'label': Feature(name='Label', feature_type='input'),
                    'label_definition': Feature(name='Label Definition', feature_type='input'),
                }
            ),
            
            # Requirements Artifact
            'artifact': Feature(
                name='Requirements Artifact', 
                feature_type='group', 
                subfeatures={
                    'specification_format': Feature(
                        name='Specification Format', 
                        feature_type='select', 
                        options=['NL', 'Constrained NL', 'Use Case', 'User Story'], 
                        multiple=True
                    ),
                    'specification_level': Feature(
                        name='Specification Level', 
                        feature_type='select', 
                        options=['High', 'Detailed'], 
                        multiple=True
                    ),
                    'stakeholder': Feature(
                        name='Stakeholder', 
                        feature_type='select', 
                        options=['End Users', 'Business Managers', 'Developers', 'Regulatory Bodies'], 
                        multiple=True
                    ),
                    'domain': Feature(name='Domain', feature_type='input', multiple=True),
                    'language': Feature(name='Language', feature_type='input', multiple=True),
                }
            ),
            
            # Generator
            'generator': Feature(
                name='Generator', 
                feature_type='group', 
                subfeatures={
                    'llm': Feature(
                        name='LLM', 
                        feature_type='select', 
                        options=[
                            'deepseek-chat', 
                            'gpt-4.1-nano-2025-04-14', 
                            'ollama/ministral-3:14b'
                        ]
                    ),
                    'temperature': Feature(name='Temperature', feature_type='input'),
                    'top_p': Feature(name='Top P', feature_type='input'),
                    'samples_per_prompt': Feature(name='Samples Per Prompt', feature_type='input'),
                    'prompt_approach': Feature(
                        name='Prompt Approach', 
                        feature_type='select', 
                        options=['Default', 'PACE']
                    ),
                    'pace_iterations': Feature(name='PACE Iterations', feature_type='input'),
                    'pace_actors': Feature(name='PACE Actors', feature_type='input'),
                    'pace_candidates': Feature(name='PACE Candidates', feature_type='input'),
                }
            ),
            
            # Output
            'output': Feature(
                name='Output', 
                feature_type='group', 
                subfeatures={
                    'output_format': Feature(
                        name='Output Format', 
                        feature_type='select', 
                        options=['JSON', 'CSV']
                    ),
                    'total_samples': Feature(name='Total Samples', feature_type='input'),
                }
            )
        } 
