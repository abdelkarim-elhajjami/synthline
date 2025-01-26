import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

class Output:
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, label: str, num_samples: int) -> str:
        """Generate a filename based on the label and number of samples"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{num_samples}samples_{timestamp}"
        return re.sub(r'[^a-zA-Z0-9_-]', '_', filename)

    def _get_output_path(self, features: Dict[str, Any], num_samples: int) -> Path:
        """Create LLM-specific directory and return full output path"""
        llm_dir = self.output_dir / features['llm']
        llm_dir.mkdir(exist_ok=True)
        
        filename = self._generate_filename(features['label'], num_samples)
        return llm_dir / filename

    def save(self, samples: List[Dict[str, Any]], format: str, features: Dict[str, Any]) -> Path:
        """Save the generated samples to a file in the specified format"""
        output_path = self._get_output_path(features, len(samples))
        output_path = output_path.with_suffix(f".{format.lower()}")
        
        if format.upper() == 'JSON':
            with open(output_path, 'w') as f:
                json.dump(samples, f, indent=2)
        else:  # CSV
            df = pd.DataFrame(samples)
            df.to_csv(output_path, index=False)
        
        return output_path