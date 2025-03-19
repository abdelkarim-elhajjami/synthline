"""
Client for interacting with LLM APIs.
Supports OpenAI and DeepSeek APIs through a unified interface.
"""
import asyncio
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI


class LLMClient:
    """
    Client for OpenAI and DeepSeek APIs.
    
    Manages connections to different LLM providers and handles request formatting,
    error handling, and response processing.
    """
    
    def __init__(self, 
                 deepseek_key: Optional[str] = None, 
                 openai_key: Optional[str] = None, 
                 logger=None):
        """
        Initialize the LLM client with API keys.
        
        Args:
            deepseek_key: API key for DeepSeek
            openai_key: API key for OpenAI
            logger: Optional logger for error reporting
        """
        self._deepseek_key = deepseek_key
        self._openai_key = openai_key
        self._deepseek_client = None
        self._openai_client = None
        self._logger = logger
        
        # Configure timeouts and retry settings
        self._request_timeout = 120  # seconds
        self._max_retries = 3

    def _get_client(self, model: str) -> AsyncOpenAI:
        """
        Return the API client for the specified model.
        
        Args:
            model: The model identifier ('deepseek-chat' or 'gpt-4o')
            
        Returns:
            An initialized AsyncOpenAI client
            
        Raises:
            ValueError: If the API key is missing for the requested model
        """
        if model == 'deepseek-chat':
            if not self._deepseek_key:
                error_msg = "DeepSeek API key is missing. Cannot use deepseek-chat model."
                if self._logger:
                    self._logger.log_error(error_msg, "llm_client", {"model": model})
                raise ValueError(error_msg)
                
            if not self._deepseek_client:
                self._deepseek_client = AsyncOpenAI(
                    api_key=self._deepseek_key,
                    base_url="https://api.deepseek.com",
                    timeout=self._request_timeout,
                    max_retries=self._max_retries
                )
            return self._deepseek_client
            
        elif model == 'gpt-4o':
            if not self._openai_key:
                error_msg = "OpenAI API key is missing. Cannot use gpt-4o model."
                if self._logger:
                    self._logger.log_error(error_msg, "llm_client", {"model": model})
                raise ValueError(error_msg)
                
            if not self._openai_client:
                self._openai_client = AsyncOpenAI(
                    api_key=self._openai_key,
                    timeout=self._request_timeout,
                    max_retries=self._max_retries
                )
            return self._openai_client
            
        else:
            raise ValueError(f"Unsupported model: {model}")

    async def _agenerate(self, 
                         prompt: str,
                         model: str,
                         temperature: float,
                         top_p: float) -> str:
        """
        Generate text from a single prompt using the specified model.
        
        Args:
            prompt: The input prompt text
            model: The model to use
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter (0-1)
            
        Returns:
            The generated completion text
            
        Raises:
            Various exceptions from the OpenAI client
        """
        client = self._get_client(model)
        
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p
            )
            
            completion_text = completion.choices[0].message.content
            
            if self._logger:
                self._logger.log_conversation(
                    prompt=prompt,
                    completion=completion_text,
                    model=model,
                    temperature=temperature,
                    top_p=top_p
                )
                
            return completion_text
            
        except Exception as e:
            error_message = str(e)
            print(f"Error in LLM call: {error_message}")
            
            if self._logger:
                self._logger.log_error(
                    error_message, 
                    "llm", 
                    {"prompt": prompt, "model": model}
                )
            # Re-raise to allow caller to handle the error
            raise

    async def _batch_generate(self,
                             prompts: List[str],
                             model: str,
                             temperature: float, 
                             top_p: float) -> List[str]:
        """
        Generate text from multiple prompts in parallel.
        
        Args:
            prompts: List of input prompts
            model: Model to use for generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            List of generated completion texts
        """
        tasks = [
            self._agenerate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                top_p=top_p
            )
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def generate(self,
                      prompts: List[str],
                      features: Dict[str, Any]) -> List[str]:
        """
        Generate text from prompts using settings from feature configuration.
        
        Args:
            prompts: List of input prompts
            features: Dictionary containing generation parameters
            
        Returns:
            List of generated completion texts
            
        Raises:
            RuntimeError: If generation fails
        """
        try:
            # Extract parameters from features
            model = features['llm']
            temperature = float(features['temperature'])
            top_p = float(features['top_p'])
            
            # Generate completions
            results = await self._batch_generate(
                prompts=prompts,
                model=model,
                temperature=temperature,
                top_p=top_p
            )
            
            # Handle any exceptions returned
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    if self._logger:
                        self._logger.log_error(
                            str(result), 
                            "llm_batch", 
                            {"model": model}
                        )
                    processed_results.append(f"[ERROR: {str(result)[:100]}...]")
                else:
                    processed_results.append(result)
                    
            return processed_results
            
        except Exception as e:
            print(f"Error in batch generation: {e}")
            if self._logger:
                self._logger.log_error(
                    str(e), 
                    "llm_batch", 
                    {"prompts": [p[:100] + "..." for p in prompts], "model": features.get('llm')}
                )
            raise RuntimeError(f"LLM generation failed: {str(e)}")