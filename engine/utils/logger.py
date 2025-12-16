"""
Logging system for Synthline.
"""
import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

class Logger:

    def __init__(self, debug_mode: bool = True):
        """
        Initialize the logger using standard logging module.
        """
        self.debug_mode = debug_mode or (os.environ.get("DEBUG_LOGGING", "false").lower() == "true")
        self.conversation_sample_rate = 0.1 # Log 10% of conversations in debug mode
        
        self._logger = logging.getLogger("Synthline")
        self._logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
        
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        
    def log_info(self, 
                 message: str, 
                 component: str, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log information to stdout.
        """
        self._log("INFO", component, {
            "message": message,
            "context": context
        })

    def log_error(self, 
                 error_msg: str, 
                 component: str, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error to stdout.
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
        """Write a structured JSON log line using standard logger."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "component": component,
            **data
        }
        json_msg = json.dumps(log_entry, ensure_ascii=False)
        
        if level == "ERROR":
            self._logger.error(json_msg)
        elif level == "DEBUG":
            self._logger.debug(json_msg)
        else:
            self._logger.info(json_msg)