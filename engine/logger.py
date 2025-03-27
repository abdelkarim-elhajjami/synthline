"""
Logging system for Synthline.
Provides error and conversation logging functionalities.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

class Logger:
    """
    Simple logger for Synthline application.
    
    Records errors and LLM conversations to organized log files.
    """
    def __init__(self, base_dir: str = "logs", debug_mode: bool = True):
        """
        Initialize the logger.
        
        Args:
            base_dir: Base directory for log files
            debug_mode: Whether to log detailed conversation data
        """
        self.log_dir = Path(base_dir)
        self.conversation_dir = self.log_dir / "conversations"
        self.error_dir = self.log_dir / "errors"
        self.debug_mode = debug_mode

    def log_conversation(self, 
                        prompt: str, 
                        completion: str, 
                        model: str, 
                        temperature: float, 
                        top_p: float) -> Optional[Path]:
        """
        Log an LLM conversation (prompt and completion).
        
        Conversation logs are only written to disk when debug_mode is True.
        
        Args:
            prompt: The input prompt
            completion: The LLM completion
            model: The model used
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Path to log file or None if debug_mode is False
        """
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
        """
        Log an error from any component.
        
        Args:
            error_msg: Error message
            component: Component name where error occurred
            context: Additional contextual information
            
        Returns:
            Path to the error log file
        """
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
            # Convert all values to strings to ensure JSON serialization
            log_data["context"] = {k: str(v) for k, v in context.items()}
        
        self._write_json(log_file, log_data)
        return log_file
    
    def log_prompt(self, 
                   prompt: str, 
                   updated_prompt: Optional[str] = None,
                   feedback: Optional[str] = None,
                   score: Optional[float] = None,
                   iteration: Optional[int] = None) -> Path:
        """
        Log prompt optimization details.
        
        Args:
            prompt: The current prompt
            updated_prompt: The updated prompt (if available)
            feedback: Critic feedback (if available)
            score: Prompt score (if available)
            iteration: Current optimization iteration
            
        Returns:
            Path to the log file
        """
        self.log_dir.mkdir(exist_ok=True, parents=True)
        prompt_dir = self.log_dir / "prompts"
        prompt_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        iter_tag = f"_iter{iteration}" if iteration is not None else ""
        
        # Determine log type based on provided data
        log_type = "prompt"
        if feedback and not updated_prompt and score is None:
            log_type = "critic"
        elif updated_prompt:
            log_type = "update"
        elif score is not None:
            log_type = "eval"
        
        # Special case for the best prompt
        if feedback == "NEW BEST PROMPT":
            log_type = "best"
        elif feedback == "FINAL OPTIMIZED PROMPT":
            log_type = "final"
        
        log_file = prompt_dir / f"{log_type}{iter_tag}_{timestamp}.json"
        
        log_data = {
            "timestamp": timestamp,
            "iteration": iteration,
            "type": log_type,
            "prompt": prompt
        }
        
        if updated_prompt is not None:
            log_data["updated_prompt"] = updated_prompt
        
        if feedback is not None:
            log_data["feedback"] = feedback
        
        if score is not None:
            log_data["score"] = float(score)
        
        self._write_json(log_file, log_data)
        return log_file
    
    def _write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Write JSON data to a file with error handling.
        
        Args:
            file_path: Path to write the file
            data: Dictionary to serialize as JSON
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing log to {file_path}: {e}")
            # We don't re-raise here to avoid crashing if logging fails