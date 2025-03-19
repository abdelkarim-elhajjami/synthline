"""
Output handler for saving generated samples.
Supports multiple output formats.
"""
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
import json
import re
import pandas as pd

class Output:
    """
    Handles saving generated data to files in various formats.
    
    Manages file naming, directory structure, and format conversion
    for generated sample data.
    """
    def __init__(self, output_dir: str = "output", logger=None):
        """
        Initialize the output handler.
        
        Args:
            output_dir: Directory to store output files
            logger: Optional logger for error reporting
        """
        self.output_dir = Path(output_dir)
        self._logger = logger
        
    def _generate_filename(self, label: str, num_samples: int) -> str:
        """
        Generate a filename based on the label and sample count.
        
        Args:
            label: Classification label for the samples
            num_samples: Number of samples in the file
            
        Returns:
            A sanitized filename string with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{label}_{num_samples}samples_{timestamp}"
        # Remove characters that might cause issues in filenames
        return re.sub(r'[^a-zA-Z0-9_-]', '_', base_filename)

    def _get_output_path(self, features: Dict[str, Any], num_samples: int) -> Path:
        """
        Get the output path for the generated data.
        
        Args:
            features: Feature configuration dictionary
            num_samples: Number of samples
            
        Returns:
            Path object for the output file (without extension)
        """
        self.output_dir.mkdir(exist_ok=True, parents=True)
        llm = features.get('llm')
        label = features.get('label')
        llm_dir = self.output_dir / llm
        llm_dir.mkdir(exist_ok=True)
        filename = self._generate_filename(label, num_samples)
        return llm_dir / filename

    def save(self, 
             samples: List[Dict[str, Any]], 
             format: str, 
             features: Dict[str, Any]) -> Path:
        """
        Save the generated samples to a file in the specified format.
        
        Args:
            samples: List of sample dictionaries to save
            format: Output format ('JSON' or 'CSV')
            features: Feature configuration dictionary
            
        Returns:
            Path to the saved file
            
        Raises:
            ValueError: If format is not supported
            IOError: If file cannot be written
        """
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
            print(error_msg)
            
            if self._logger:
                self._logger.log_error(
                    error_msg, 
                    "output", 
                    {"format": format, "output_path": str(output_path)}
                )
            raise
    
    def _save_json(self, samples: List[Dict[str, Any]], path: Path) -> None:
        """
        Save data in JSON format.
        
        Args:
            samples: List of sample dictionaries
            path: Output file path
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
    
    def _save_csv(self, samples: List[Dict[str, Any]], path: Path) -> None:
        """
        Save data in CSV format.
        
        Args:
            samples: List of sample dictionaries
            path: Output file path
        """
        pd.DataFrame(samples).to_csv(path, index=False, encoding='utf-8')