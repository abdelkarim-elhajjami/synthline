"""
Logging system for Synthline.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

class Logger:
    """Logger for Synthline."""
    def __init__(self, base_dir: str = "logs", debug_mode: bool = True):
        """Initialize the logger."""
        
        self.log_dir = Path(base_dir)
        self.conversation_dir = self.log_dir / "conversations"
        self.error_dir = self.log_dir / "errors"
        self.pace_dir = self.log_dir / "pace"
        self.debug_mode = debug_mode

    def log_conversation(self, 
                        prompt: str, 
                        completion: str, 
                        model: str, 
                        temperature: float, 
                        top_p: float) -> Optional[Path]:
        """Log an LLM conversation (prompt and completion)."""
        
        if not self.debug_mode:
            return None
            
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.conversation_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = self.conversation_dir / f"conversation_{timestamp}.json"
        
        log_data = {
            "timestamp": timestamp,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "prompt": prompt,
            "completion": completion
        }
        
        self._write_json(log_file, log_data)
        return log_file
    
    def log_error(self, 
                 error_msg: str, 
                 component: str, 
                 context: Optional[Dict[str, Any]] = None) -> Path:
        """Log an error from any component."""
        
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.error_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = self.error_dir / f"{component}_error_{timestamp}.json"
        
        log_data = {
            "timestamp": timestamp,
            "component": component,
            "error": error_msg
        }
        
        if context:
            log_data["context"] = {k: str(v) for k, v in context.items()}
        
        self._write_json(log_file, log_data)
        return log_file
    
    def log_prompt(self, 
                  prompt: str,
                  score: Optional[float] = None,
                  event: Optional[str] = None) -> Optional[Path]:
        """Log prompt optimization events."""
        
        if not self.debug_mode:
            return None

        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.pace_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        if event == "NEW BEST PROMPT":
            log_type = "update"
        elif event == "FINAL OPTIMIZED PROMPT":
            log_type = "final"
        
        log_file = self.pace_dir / f"{log_type}_{timestamp}.json"
        
        log_data = {
            "timestamp": timestamp,
            "event": event,
            "prompt": prompt,
            "score": score
        }
        
        self._write_json(log_file, log_data)
        return log_file
    
    def _write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write JSON data to a file with error handling."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing log to {file_path}: {e}") 