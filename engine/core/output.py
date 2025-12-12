"""
Output handler for saving generated data.
Supports both JSON and CSV output formats.
"""
from pathlib import Path
from typing import Any, Dict, List
import json
import pandas as pd
from utils.logger import Logger

class Output:
    """
    Stateless output handler.
    Processes generated data into strings (JSON/CSV) for direct download.
    Does NOT save to disk.
    """
    def __init__(self, logger: Logger):
        self._logger = logger

    def process(self, samples: List[Dict[str, Any]], format: str) -> str:
        """
        Process samples into a string format (JSON or CSV).
        This content is sent directly to the client for download.
        """
        format = format.upper()
        
        if format not in ['JSON', 'CSV']:
            raise ValueError(f"Unsupported output format: {format}")
        
        try:
            if format == 'JSON':
                return json.dumps(samples, indent=2, ensure_ascii=False)
            elif format == 'CSV':
                return pd.DataFrame(samples).to_csv(index=False, encoding='utf-8')
            return ""
        except Exception as e:
            self._logger.log_error(f"Error processing output: {e}", "output")
            raise
    
    def _save_csv(self, samples: List[Dict[str, Any]], path: Path) -> None:
        """Save data in CSV format."""
        pd.DataFrame(samples).to_csv(path, index=False, encoding='utf-8')