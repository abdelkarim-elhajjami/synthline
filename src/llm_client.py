import asyncio
from openai import AsyncOpenAI
from typing import Any, Dict, List

class LLMClient:
    """Client for interacting with OpenAI and DeepSeek APIs."""

    def __init__(self, deepseek_key: str, openai_key: str):
        self._deepseek_key = deepseek_key
        self._openai_key = openai_key
        self._deepseek_client = None
        self._openai_client = None

    def _get_client(self, model: str) -> AsyncOpenAI:
        """Get the appropriate client based on the model type."""
        if model == 'deepseek-reasoner':
            if not self._deepseek_client:
                self._deepseek_client = AsyncOpenAI(
                    api_key=self._deepseek_key,
                    base_url="https://api.deepseek.com",
                    timeout=120,
                    max_retries=3
                )
            return self._deepseek_client
        else:  # gpt-4o
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
            return completion.choices[0].message.content or ""
        except Exception as e:
            print(f"Error in agenerate: {e}")
            return ""

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
            return [""] * len(prompts)