"""
Output handler for saving generated data.
Supports both JSON and CSV output formats.
"""
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
import json
import re
import pandas as pd
from utils.logger import Logger

class Output:
    """Handles saving generated data to files in various formats."""
    def __init__(self, logger: Logger, output_dir: str = "output"):
        """Initialize the output handler."""
        self.output_dir = Path(output_dir)
        self._logger = logger
        
    def _generate_filename(self, label: str, num_samples: int) -> str:
        """Generate a filename based on the label and sample count."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{label}_{num_samples}samples_{timestamp}"
        return re.sub(r'[^a-zA-Z0-9_-]', '_', base_filename)

    def _get_output_path(self, features: Dict[str, Any], num_samples: int) -> Path:
        """Get the output path for the generated data."""
        self.output_dir.mkdir(exist_ok=True, parents=True)
        llm = features['llm']
        llm_name = llm.split('/')[-1] if llm.startswith('ollama/') else llm
        label = features['label']
        llm_dir = self.output_dir / llm_name
        llm_dir.mkdir(exist_ok=True, parents=True)
        filename = self._generate_filename(label, num_samples)
        return llm_dir / filename

    def save(self, 
             samples: List[Dict[str, Any]], 
             format: str, 
             features: Dict[str, Any]) -> Path:
        """Save the generated data to a file in the specified format."""
        format = format.upper()
        
        if format not in ['JSON', 'CSV']:
            raise ValueError(f"Unsupported output format: {format}")
          
        output_path = self._get_output_path(features, len(samples))
        output_path = output_path.with_suffix(f".{format.lower()}")
        
        try:
            if format == 'JSON':
                self._save_json(samples, output_path)
            elif format == 'CSV':
                self._save_csv(samples, output_path)
                
            return output_path
            
        except Exception as e:
            error_msg = f"Error saving output to {output_path}: {e}"
            if self._logger:
                self._logger.log_error(
                    error_msg, 
                    "output", 
                    {"format": format, "output_path": str(output_path)}
                )
            raise
    
    def _save_json(self, samples: List[Dict[str, Any]], path: Path) -> None:
        """Save data in JSON format."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
    
    def _save_csv(self, samples: List[Dict[str, Any]], path: Path) -> None:
        """Save data in CSV format."""
        pd.DataFrame(samples).to_csv(path, index=False, encoding='utf-8')