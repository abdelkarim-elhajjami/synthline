from typing import Dict, Any, List
from itertools import product
from llm_client import LLMClient
from promptline import Promptline

class Generator:
    """Generator class for creating synthetic data."""

    def __init__(self, llm: LLMClient, promptline: Promptline, batch_size: int = 20):
        self._llm = llm
        self._promptline = promptline
        self._batch_size = batch_size
        self._multiple_select_features = [
            "requirement_type",
            "specification_format",
            "specification_level",
            "requirement_source",
            "domain",
            "language",
        ]

    def _flatten_configurations(self, feature_values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build a list of atomic configurations (one for each combination)
        """
        configs_by_feature = {}
        for feature in self._multiple_select_features:
            value = feature_values.get(feature)
            if isinstance(value, list):
                configs_by_feature[feature] = value
            else:
                configs_by_feature[feature] = [value]

        all_configurations = []
        feature_names = list(configs_by_feature.keys())
        product_of_values = product(*(configs_by_feature[f] for f in feature_names))
        
        for tuple_of_values in product_of_values:
            atomic_config = feature_values.copy()
            for feature_name, selected_value in zip(feature_names, tuple_of_values):
                if selected_value is not None:
                    atomic_config[feature_name] = selected_value
            all_configurations.append(atomic_config)
        
        return all_configurations

    def generate_samples(
        self, 
        subset_size: int, 
        feature_values: Dict[str, Any], 
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        Generate exactly subset_size data points, distributing them evenly
        among all atomic configurations.
        """
        print("Feature values received:", feature_values)

        configurations = self._flatten_configurations(feature_values)
        n_configs = len(configurations)
        
        base_count = subset_size // n_configs
        remainder = subset_size % n_configs

        all_samples = []
        current_i = 0

        for i, config in enumerate(configurations):
            count_for_this_config = base_count + (1 if i < remainder else 0)

            for _ in range(count_for_this_config):
                prompt = self._promptline.build(config)
                try:
                    response_list = self._llm.generate([prompt], config)
                    generation = response_list[0] if response_list else ""
                    
                    if generation.strip():
                        sample = {
                            "requirement_text": generation.strip(),
                            "requirement_type": config.get("requirement_type"),
                            "specification_format": config.get("specification_format"),
                            "specification_level": config.get("specification_level"),
                            "requirement_source": config.get("requirement_source"),
                            "domain": config.get("domain"),
                            "language": config.get("language"),
                            "label": config.get("label"),
                        }
                        all_samples.append(sample)

                except Exception as e:
                    print(f"Error generating from configuration {config}: {e}")

                current_i += 1
                if progress_callback:
                    progress_callback(current_i / subset_size * 100)

        print(f"Generated {len(all_samples)} samples in total.")
        return all_samples