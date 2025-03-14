from typing import List, Dict, Optional

class Feature:
    """
    Represents a configurable feature with its properties and sub-features.

    Attributes:
        name (str): The name of the feature
        feature_type (str): The type of the feature (e.g., 'group', 'select', 'input')
        options (List[str]): Available options for select-type features
        multiple (bool): Whether multiple selections are allowed
        subfeatures (Dict[str, 'Feature']): Nested features
    """
    def __init__(self, name: str, feature_type: str, 
                 options: Optional[List[str]] = None, 
                 multiple: bool = False, 
                 subfeatures: Optional[Dict[str, 'Feature']] = None):
        self.name = name
        self.feature_type = feature_type
        self.options = options or []
        self.multiple = multiple
        self.subfeatures = subfeatures or {}

class FM:
    """
    Feature Manager for Synthline.
    Manages the hierarchical structure of configurable features.
    """
    def __init__(self) -> None:
        self.features: Dict[str, Feature] = {
            # 1. Generator
            'generator': Feature('Generator', 'group', subfeatures={
                'llm': Feature('LLM', 'select', ['deepseek-chat', 'gpt-4o']),
                'temperature': Feature('Temperature (0-2)', 'input'),
                'top_p': Feature('Top P (0-1)', 'input'),
                'samples_per_call': Feature('Samples Per API Call', 'input'),
            }),
            
            # 2. Artifact
            'artifact': Feature('Artifact', 'select', ['Requirements'], subfeatures={
                'requirements': Feature('Requirements', 'group', subfeatures={
                    'specification_format': Feature('Specification Format', 'select', [
                        'NL', 'ConstrainedNL', 'UseCase', 'UserStory'
                    ], multiple=True),
                    'specification_level': Feature('Specification Level', 'select', [
                        'HighLevelSpecification', 'DetailedSpecification'
                    ], multiple=True),
                    'requirement_source': Feature('Requirement Source', 'select', [
                        'EndUsers', 'BusinessManagers', 'DevelopementTeam', 'RegulatoryBodies'
                    ], multiple=True),
                    'domain': Feature('Domain/s', 'input'),
                    'language': Feature('Language/s', 'input'),
                })
            }),
            
            # 3. ML Task
            'ml_task': Feature('ML Task', 'select', ['Classification'], subfeatures={
                'classification': Feature('Classification', 'group', subfeatures={
                    'label': Feature('Label', 'input'),
                    'label_description': Feature('Label Description', 'input'),
                })
            }),
            
            # 4. Output
            'output': Feature('Output', 'group', subfeatures={
                'output_format': Feature('Output Format', 'select', ['JSON', 'CSV']),
                'subset_size': Feature('Subset Size', 'input'),
            })
        }
        