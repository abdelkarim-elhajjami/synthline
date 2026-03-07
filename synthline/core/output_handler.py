"""Output handler for generated data."""
from typing import Any, Dict, List

import pandas as pd
from synthline.core.constants import OPERATING_FIELDS
from synthline.utils.logger import Logger

class OutputHandler:
    def __init__(self, logger: Logger):
        self._logger = logger

    def to_csv(self, samples: List[Dict[str, Any]]) -> str:
        try:
            return pd.DataFrame(samples).to_csv(index=False, encoding='utf-8')
        except Exception as e:
            self._logger.log_error(f"Error processing CSV output: {e}", "output")
            raise

    def format_sample(self, sample_text: str, atomic_config: Dict[str, Any]) -> Dict[str, Any]:
        """Format the raw sample text and atomic config into a structured dictionary."""
        sample = {"Text": sample_text}

        cls_label = atomic_config.get("classification_label", "")
        cls_def = atomic_config.get("classification_label_def", "")
        if cls_label:
            sample["Label"] = cls_label
        if cls_def:
            sample["LabelDefinition"] = cls_def

        for k, v in atomic_config.items():
            if k not in OPERATING_FIELDS:
                # Use only the leaf name from dotted FM paths for cleaner CSV columns
                # e.g. "Root.Category.Subtype" -> "Subtype"
                col_name = k.rsplit(".", 1)[-1] if "." in k else k
                sample[col_name] = v

        return sample
