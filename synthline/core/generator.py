import math
import random
from typing import Any, Dict, List, Optional, Tuple
from synthline.core.llm import LLMClient
from synthline.core.promptline import Promptline
from synthline.core.schemas import samples_schema
from synthline.types import GenerationResult
from synthline.utils.logger import Logger
from synthline.utils.parsing import parse_completion_with_meta
from synthline.utils.progress import ProgressFn, track_progress


class Generator:
    def __init__(
        self,
        llm: LLMClient,
        promptline: Promptline,
        logger: Logger
    ) -> None:
        self._llm = llm
        self._promptline = promptline
        self._logger = logger

    async def generate(
        self,
        features: Dict[str, Any],
        progress_callback: ProgressFn = None,
        api_keys: Optional[Dict[str, str]] = None
    ) -> GenerationResult:
        all_samples = []
        fewer_samples_received = False
        parsing_degraded = False

        total_samples = int(features['total_samples'])
        samples_per_prompt = int(features['samples_per_prompt'])

        llm_settings = {k: features[k] for k in ['llm', 'temperature', 'top_p']}

        if 'optimized_atomic_prompts' in features:
            atomic_configs = []
            for optimized_prompt_data in features['optimized_atomic_prompts']:
                config = {**llm_settings, **optimized_prompt_data['config']}
                config['optimized_prompt'] = optimized_prompt_data['optimized_prompt']
                if 'pace_score' in optimized_prompt_data:
                    config['pace_score'] = optimized_prompt_data['pace_score']
                atomic_configs.append(config)
        else:
            atomic_configs = self._promptline.get_atomic_configurations(features)

            for config in atomic_configs:
                config.update(llm_settings)

        n_configs = len(atomic_configs)
        active_configs, sample_counts = self._distribute_samples(
            total_samples, n_configs, samples_per_prompt, atomic_configs
        )

        generated_so_far = 0

        for i, config in enumerate(active_configs):
            samples_for_config = sample_counts[i]

            config_samples = []
            retries = 0
            max_retries = 3
            max_calls = (samples_for_config // max(1, samples_per_prompt)) + max_retries + 1
            calls = 0
            while len(config_samples) < samples_for_config and calls < max_calls:
                calls += 1
                samples_needed = samples_for_config - len(config_samples)
                request_count = min(samples_needed, samples_per_prompt)

                try:
                    new_samples, received_count, call_degraded, parse_method = (
                        await self._generate_samples(
                            atomic_config=config,
                            request_count=request_count,
                            api_keys=api_keys,
                        )
                    )
                except Exception:
                    fewer_samples_received = True
                    break

                if call_degraded:
                    parsing_degraded = True

                # Retry only on actual parse failure (malformed output)
                if call_degraded and received_count < request_count:
                    retries += 1
                    if retries >= max_retries:
                        fewer_samples_received = True
                        self._logger.log_warning(
                            f"Parse failed {retries}x, skipping config "
                            f"({len(config_samples)}/{samples_for_config} collected).",
                            "generator",
                        )
                        break
                    self._logger.log_warning(
                        f"Got {received_count}/{request_count} samples "
                        f"({parse_method}), retrying ({retries}/{max_retries}).",
                        "generator",
                    )
                    continue

                # Accept valid samples (loop fills any gap on next iteration)
                if new_samples:
                    config_samples.extend(new_samples)
                    generated_so_far += len(new_samples)
                    if progress_callback and total_samples > 0:
                        progress = min(100.0, (generated_so_far / total_samples) * 100.0)
                        await track_progress(progress_callback, progress)
                else:
                    break

            all_samples.extend(config_samples)

        all_samples = all_samples[:total_samples]

        if progress_callback:
            await track_progress(progress_callback, 100)

        return GenerationResult(
            samples=all_samples,
            fewer_samples_received=fewer_samples_received,
            parsing_degraded=parsing_degraded,
        )

    async def generate_for_configs(
        self,
        config_requests: List[Tuple[Dict[str, Any], int]],
        samples_per_prompt: int,
        api_keys: Optional[Dict[str, str]] = None,
    ) -> GenerationResult:
        """Generate samples for explicit (config, count) pairs.

        Unlike :meth:`generate`, this method does **not** distribute
        samples via ``_distribute_samples``.  Each ``(config, count)``
        pair is fulfilled independently using the exact config provided.

        Used by the verification loop for config-aware regeneration so
        that rejected samples are replaced from the *same* configuration.
        """
        all_samples: List[Dict[str, Any]] = []
        fewer_samples_received = False
        parsing_degraded = False

        for config, target_count in config_requests:
            # Ensure the exact same prompt is reused (bypass promptline.build).
            regen_config = {k: v for k, v in config.items() if k != "prompt"}
            if "optimized_prompt" not in regen_config and "prompt" in config:
                regen_config["optimized_prompt"] = config["prompt"]

            config_samples: List[Dict[str, Any]] = []
            retries = 0
            max_retries = 3
            max_calls = math.ceil(target_count / max(1, samples_per_prompt)) + max_retries + 1
            calls = 0

            while len(config_samples) < target_count and calls < max_calls:
                calls += 1
                # Always request samples_per_prompt to keep generation
                # conditions identical to the original run (same prompt
                # text, same JSON schema).  Excess samples are trimmed.
                request_count = samples_per_prompt

                try:
                    new_samples, received_count, call_degraded, parse_method = (
                        await self._generate_samples(
                            atomic_config=regen_config,
                            request_count=request_count,
                            api_keys=api_keys,
                        )
                    )
                except Exception:
                    fewer_samples_received = True
                    break

                if call_degraded:
                    parsing_degraded = True

                if call_degraded and received_count < request_count:
                    retries += 1
                    if retries >= max_retries:
                        fewer_samples_received = True
                        self._logger.log_warning(
                            f"[regen] Parse failed {retries}x, skipping config "
                            f"({len(config_samples)}/{target_count} collected).",
                            "generator",
                        )
                        break
                    continue

                if new_samples:
                    config_samples.extend(new_samples)
                else:
                    break

            all_samples.extend(config_samples[:target_count])

        return GenerationResult(
            samples=all_samples,
            fewer_samples_received=fewer_samples_received,
            parsing_degraded=parsing_degraded,
        )

    async def _generate_samples(
        self,
        atomic_config: Dict[str, Any],
        request_count: int,
        api_keys: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Dict[str, Any]], int, bool, str]:
        """Generate samples from a single LLM call.

        Args:
            atomic_config: Feature configuration for this generation.
            request_count: Exact number of samples to request (schema + prompt).
            api_keys: Optional API keys.

        Returns:
            (samples, valid_count, parsing_degraded, parse_method)
        """
        new_samples = []
        sample_texts = []
        parsing_degraded = False
        parse_method = "plaintext"

        response_format = samples_schema(request_count)

        if 'optimized_prompt' in atomic_config:
            prompt = atomic_config['optimized_prompt']
        else:
            prompt = self._promptline.build(atomic_config, samples_per_prompt=request_count)

        try:
            completion_list = await self._llm.get_batch_completions(
                prompts=[prompt],
                features=atomic_config,
                api_keys=api_keys,
                response_format=response_format,
            )
            completion = completion_list[0]

            sample_texts, parsing_degraded, parse_method = parse_completion_with_meta(completion, request_count)

            for sample_text in sample_texts[:request_count]:
                if sample_text and sample_text.strip():
                    new_samples.append({
                        "text": sample_text.strip(),
                        "config": {**atomic_config, "prompt": prompt},
                    })

        except Exception as e:
            self._logger.log_error(
                f"Generation error: {e}",
                "generator",
                {"atomic_config": atomic_config},
            )
            raise

        valid_count = sum(1 for s in sample_texts if s and s.strip())
        return new_samples, valid_count, parsing_degraded, parse_method

    def _distribute_samples(
        self,
        total_samples: int,
        n_configs: int,
        samples_per_prompt: int,
        atomic_configs: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Select configurations and distribute samples across them.

        The number of required calls is ``k = ceil(total_samples / samples_per_prompt)``.

        * When *k <= n_configs*, a random subset of *k* configurations is
          selected (one call per config).
        * When *k > n_configs*, all configurations are used and extra calls
          are distributed evenly across them.

        Returns ``(selected_configs, sample_counts)`` aligned by index.
        """
        if n_configs == 0:
            return [], []

        k = math.ceil(total_samples / samples_per_prompt)

        if k <= n_configs:
            indices = random.sample(range(n_configs), k)
            selected = [atomic_configs[i] for i in indices]
            counts = [samples_per_prompt] * k
            return selected, counts
        else:
            calls_base = k // n_configs
            calls_remainder = k % n_configs
            counts = []
            for i in range(n_configs):
                calls = calls_base + (1 if i < calls_remainder else 0)
                counts.append(calls * samples_per_prompt)
            return list(atomic_configs), counts
