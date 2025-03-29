"""
Shared text and JSON parsing utilities for LLM outputs.
"""
import json
from typing import List

def parse_completion(text: str, expected_count: int) -> List[str]:
    """
    Parse samples from LLM completion text.
    
    Args:
        text: LLM completion text
        expected_count: Expected number of samples
        
    Returns:
        List of sample texts
    """
    if expected_count > 1:
        samples = try_parse_json_array(text)
        if samples:
            return samples
    
    return [text.strip()]

def try_parse_json_array(text: str) -> List[str]:
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