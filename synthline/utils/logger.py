"""
Logging system for Synthline.
"""
import json
import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from synthline.core.constants import OPERATING_FIELDS, PACE_EVENTS


class Logger:
    CONVERSATION_SAMPLE_RATE = 0.1

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.conversation_sample_rate = self.CONVERSATION_SAMPLE_RATE

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
        self._log("INFO", component, {
            "message": message,
            "context": context
        })

    def log_warning(self,
                   message: str,
                   component: str,
                   context: Optional[Dict[str, Any]] = None) -> None:
        self._log("WARNING", component, {
            "message": message,
            "context": context
        })

    def log_error(self,
                 message: str,
                 component: str,
                 context: Optional[Dict[str, Any]] = None) -> None:
        self._log("ERROR", component, {
            "message": message,
            "context": context
        })

    def log_prompt(self,
                  prompt: str,
                  score: float,
                  event: str,
                  config: Dict[str, Any]) -> None:
        if not self.debug_mode:
            return

        if event not in PACE_EVENTS:
            return

        config_preview = {
            k: v for k, v in config.items()
            if k not in OPERATING_FIELDS and not str(k).startswith("__")
        }
        self._log("INFO", "PACE", {
            "event": event,
            "score": score,
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "config": config_preview,
        })

    def log_conversation(self,
                        prompt: str,
                        completion: str,
                        model: str,
                        temperature: float,
                        top_p: float) -> None:
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "component": component,
            **data
        }
        json_msg = json.dumps(log_entry, ensure_ascii=False, default=str)

        if level == "ERROR":
            self._logger.error(json_msg)
        elif level == "WARNING":
            self._logger.warning(json_msg)
        elif level == "DEBUG":
            self._logger.debug(json_msg)
        else:
            self._logger.info(json_msg)
