"""
Generator for synthetic data samples.
Handles LLM generation and output parsing.
"""
import asyncio
import json
import re
from typing import Any, Dict, List, Tuple, Optional, Callable, Awaitable

from llm_client import LLMClient
from promptline import Promptline

# Type for progress callback
ProgressCallback = Optional[Callable[[float], Awaitable[None]]]

class Generator:
    """
    Generator for creating synthetic data samples using LLMs.
    
    Handles the process of generating samples based on feature configurations,
    parsing LLM outputs, and tracking progress.
    """
    
    def __init__(
        self, 
        llm: LLMClient, 
        promptline: Promptline, 
        batch_size: int,
        logger=None
    ) -> None:
        """
        Initialize the generator.
        
        Args:
            llm: LLM client for text generation
            promptline: Prompt template manager
            batch_size: Number of requests to process in parallel
            logger: Optional logger for error reporting
        """
        self._llm = llm
        self._promptline = promptline
        self._batch_size = batch_size
        self._logger = logger
        self._fewer_samples_received = False

    async def generate_samples(
        self,
        feature_values: Dict[str, Any],
        progress_callback: Optional[Callable[[float], Awaitable[None]]] = None,
        optimized_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic requirements based on feature configuration.
        
        Args:
            feature_values: Dictionary of feature settings
            progress_callback: Optional callback for reporting progress
            optimized_prompt: Optional optimized prompt from PACE
            
        Returns:
            List of generated samples as dictionaries
        """
        if optimized_prompt:
            feature_values = {**feature_values, 'optimized_prompt': optimized_prompt}
        
        all_samples = []
        self._fewer_samples_received = False
        
        total_samples = int(feature_values.get('total_samples'))
        samples_per_prompt = int(feature_values.get('samples_per_prompt'))
        
        # Generate samples until we have enough
        while len(all_samples) < total_samples:
            samples_needed = total_samples - len(all_samples)
            request_count = min(samples_needed, samples_per_prompt)
            
            # Generate samples
            new_samples, received_count = await self._generate_for_config(
                config=feature_values,
                samples_needed=samples_needed,
                samples_per_prompt=request_count
            )
            
            # Check if we received fewer samples than requested (token limit)
            if received_count < request_count and received_count > 0:
                self._fewer_samples_received = True
                
                # Log the issue
                if self._logger:
                    self._logger.log_error(
                        f"Received fewer samples than requested ({received_count}/{request_count}), likely due to output token limit.",
                        "generator",
                        {"config": feature_values}
                    )
            
            # Add new samples to our collection
            if new_samples:
                all_samples.extend(new_samples)
                
                # Update progress if callback provided
                if progress_callback:
                    progress = min(100, (len(all_samples) / total_samples) * 100)
                    # Call the callback asynchronously
                    if asyncio.iscoroutinefunction(progress_callback):
                        await progress_callback(progress)
                    else:
                        await asyncio.to_thread(progress_callback, progress)
            else:
                # If we didn't get any new samples, break to avoid infinite loop
                break
            
        # Update progress to 100% when done
        if progress_callback:
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(100)
            else:
                await asyncio.to_thread(progress_callback, 100)
            
        return all_samples
    
    async def _generate_for_config(
        self, 
        config: Dict[str, Any], 
        samples_needed: int, 
        samples_per_prompt: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Generate samples for a specific configuration.
        
        Args:
            config: Feature configuration dictionary
            samples_needed: Number of samples needed
            samples_per_prompt: Number of samples to request per prompt
            
        Returns:
            Tuple of (list of sample dictionaries, count of samples received)
        """
        new_samples = []
        sample_texts = []
        
        # Check if we have an optimized prompt
        optimized_prompt = config.get('optimized_prompt')
        if optimized_prompt:
            prompt = optimized_prompt
        else:
            prompt = self._promptline.build(config, samples_per_prompt)
        
        try:
            completion_list = await self._llm.generate([prompt], config)
            generation = completion_list[0] if completion_list else ""
            
            if not generation.strip():
                print(f"Warning: Empty completion received")
                return [], 0
                
            # Parse the completion to extract requirements
            sample_texts = self._parse_json_samples(generation, samples_per_prompt)
            
            # Create samples from the extracted texts
            for sample_text in sample_texts[:samples_needed]:
                if sample_text and sample_text.strip():
                    new_samples.append(self._create_sample(sample_text.strip(), config))
        
        except Exception as e:
            error_msg = f"Error generating from configuration: {e}"
            print(error_msg)
            if self._logger:
                self._logger.log_error(error_msg, "generator", config)
        
        return new_samples, len(sample_texts)

    def _create_sample(self, sample_text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a sample dictionary with metadata from config.
        
        Args:
            sample_text: The generated sample text
            config: Feature configuration
            
        Returns:
            Sample dictionary with text and metadata
        """
        return {
            "text": sample_text,
            "label": config.get("label"),
            "domain": config.get("domain"),
            "language": config.get("language"),
            "stakeholder": config.get("stakeholder"),
            "specification_format": config.get("specification_format"),
            "specification_level": config.get("specification_level")
        }
    
    def _parse_json_samples(self, text: str, expected_count: int) -> List[str]:
        """
        Parse samples from LLM completion text.
        
        Uses multiple strategies to parse JSON arrays or plain text
        based on the expected format.
        
        Args:
            text: LLM completion text
            expected_count: Expected number of samples
            
        Returns:
            List of sample texts
        """
        # First try: Extract and parse JSON array
        samples = self._try_parse_json_array(text)
        if samples:
            return samples
        
        # Second try: Handle single sample case
        if expected_count == 1:
            return [text.strip()]
        
        # Fallback: Extract line by line
        return self._extract_samples_from_lines(text)
    
    def _try_parse_json_array(self, text: str) -> List[str]:
        """
        Try to extract a JSON array from text.
        
        Args:
            text: Text that may contain a JSON array
            
        Returns:
            List of strings from the array or empty list if parsing fails
        """
        json_start = text.find('[')
        json_end = text.rfind(']')
        
        if json_start < 0 or json_end <= json_start:
            return []
        
        json_text = text[json_start:json_end+1]
        
        # Try standard JSON parsing
        try:
            data = json.loads(json_text)
            if isinstance(data, list):
                return [item.strip() for item in data if isinstance(item, str) and item.strip()]
        except json.JSONDecodeError:
            pass
        
        # Try with common fixes for JSON formatting issues
        try:
            cleaned_text = json_text.replace('\\"', '"').replace('""', '"')
            data = json.loads(cleaned_text)
            if isinstance(data, list):
                return [item.strip() for item in data if isinstance(item, str) and item.strip()]
        except:
            pass
        
        return []
    
    def _extract_samples_from_lines(self, text: str) -> List[str]:
        """
        Extract samples by splitting text into lines and cleaning them.
        
        Args:
            text: Multi-line text to parse
            
        Returns:
            List of extracted sample texts
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        samples = []
        
        for line in lines:
            # Skip JSON syntax and look for substantial content
            if not line.startswith(('[', ']', '{', '}')) and len(line) > 10:
                # Remove numbering, quotation marks, and other formatting
                cleaned = re.sub(r'^\d+\.|\-\s+|^"|"$|,$', '', line).strip()
                if cleaned:
                    samples.append(cleaned)
        
        # If we couldn't extract anything, return the whole text as one sample
        return samples if samples else [text.strip()]
    