"""
Logging system for Synthline.
"""
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

class Logger:
    """
    Stateless logger for Synthline.
    Logs to stdout in structured JSON format.
    Controlled by DEBUG_LOGGING environment variable.
    """
    def __init__(self, base_dir: str = "", debug_mode: bool = True):
        """
        Initialize the logger.
        base_dir is ignored (kept for compatibility).
        debug_mode is read from env var 'DEBUG_LOGGING' if not provided explicitly.
        """
        self.debug_mode = debug_mode or (os.environ.get("DEBUG_LOGGING", "false").lower() == "true")
        self.conversation_sample_rate = 0.1 # Log 10% of conversations in debug mode
        
    def log_error(self, 
                 error_msg: str, 
                 component: str, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error to stdout.
        Errors are ALWAYS logged, regardless of debug mode.
        """
        self._log("ERROR", component, {
            "error": error_msg,
            "context": context
        })
    
    def log_prompt(self, 
                  prompt: str,
                  score: float,
                  event: str,
                  config: Dict[str, Any]) -> None:
        """
        Log prompt optimization events.
        Only logs 'NEW BEST PROMPT' or 'FINAL' events to avoid noise.
        Only active in DEBUG_MODE.
        """
        if not self.debug_mode:
            return
            
        if event not in ["NEW BEST PROMPT", "FINAL OPTIMIZED PROMPT"]:
            return

        self._log("INFO", "PACE", {
            "event": event,
            "score": score,
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "config": {
                k: v for k, v in config.items()
                if k in ['label', 'specification_format', 'stakeholder']
            }
        })
    
    def log_conversation(self, 
                        prompt: str, 
                        completion: str, 
                        model: str, 
                        temperature: float, 
                        top_p: float) -> None:
        """
        Log an LLM conversation.
        Only active in DEBUG_MODE.
        Uses sampling to log only a fraction of conversations to avoid spam.
        """
        if not self.debug_mode:
            return

        if random.random() > self.conversation_sample_rate:
            return
            
        self._log("DEBUG", "LLM", {
            "model": model,
            "temperature": temperature,
            "prompt": prompt,
            "completion": completion
        })

    def _log(self, level: str, component: str, data: Dict[str, Any]) -> None:
        """Write a structured JSON log line to stdout."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "component": component,
            **data
        }
        print(json.dumps(log_entry, ensure_ascii=False))