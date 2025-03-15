import asyncio
from openai import AsyncOpenAI
from typing import Any, Dict, List

class LLMClient:
    """Client for OpenAI and DeepSeek APIs."""
    
    def __init__(self, deepseek_key: str, openai_key: str, logger=None):
        self._deepseek_key = deepseek_key
        self._openai_key = openai_key
        self._deepseek_client = None
        self._openai_client = None
        self._logger = logger

    def _get_client(self, model: str) -> AsyncOpenAI:
        """Return the API client for the specified model."""
        if model == 'deepseek-chat':
            if not self._deepseek_client:
                self._deepseek_client = AsyncOpenAI(
                    api_key=self._deepseek_key,
                    base_url="https://api.deepseek.com",
                    timeout=120,
                    max_retries=3
                )
            return self._deepseek_client
        elif model == 'gpt-4o':
            if not self._openai_client:
                self._openai_client = AsyncOpenAI(
                    api_key=self._openai_key,
                    timeout=120,
                    max_retries=3
                )
            return self._openai_client

    async def _agenerate(
        self, 
        prompt: str,
        model: str,
        temperature: float,
        top_p: float
    ) -> str:
        client = self._get_client(model)
        
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p
            )
            
            completion = completion.choices[0].message.content
            
            if self._logger:
                self._logger.log_conversation(
                    prompt=prompt,
                    completion=completion,
                    model=model,
                    temperature=temperature,
                    top_p=top_p
                )
                
            return completion
            
        except Exception as e:
            error_message = str(e)
            print(f"Error in LLM call: {error_message}")
            
            if self._logger:
                self._logger.log_error(
                    error_message, 
                    "llm", 
                    {"prompt": prompt, "model": model}
                )
            return f"[ERROR: {error_message[:100]}...]"

    async def _batch_generate(
        self,
        prompts: List[str],
        model: str,
        temperature: float, 
        top_p: float
    ) -> List[str]:
        calls = [
            self._agenerate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                top_p=top_p
            )
            for prompt in prompts
        ]
        return await asyncio.gather(*calls)

    def generate(
        self,
        prompts: List[str],
        features: Dict[str, Any]
    ) -> List[str]:
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            samples = loop.run_until_complete(
                self._batch_generate(
                    prompts=prompts,
                    model=features['llm'],
                    temperature=float(features['temperature']),
                    top_p=float(features['top_p'])
                )
            )
            return samples
            
        except Exception as e:
            print(f"Error in batch generation: {e}")
            if self._logger:
                self._logger.log_error(
                    str(e), 
                    "llm_batch", 
                    {"prompts": [p[:100] + "..." for p in prompts], "model": features.get('llm')}
                )
            raise RuntimeError(f"LLM generation failed: {str(e)}")      