"""
Shared text and JSON parsing utilities for LLM outputs.
"""
import json
import re
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
        # First try parsing as valid JSON array
        samples = try_parse_json_array(text)
        if samples:
            return samples
            
        # If that fails, try more aggressive extraction techniques
        samples = try_extract_samples_with_regex(text)
        if samples:
            return samples
    
    # If all parsing attempts fail or expected_count is 1, return the whole text
    return [text.strip()]

def try_parse_json_array(text: str) -> List[str]:
    """
    Try to extract a JSON array from text.
    
    Args:
        text: Text that may contain a JSON array
        
    Returns:
        List of strings from the array or empty list if parsing fails
    """
    # Look for the outermost array brackets
    json_start = text.find('[')
    json_end = text.rfind(']')
    
    if json_start < 0 or json_end <= json_start:
        return []
    
    json_text = text[json_start:json_end+1]
    
    # Try multiple parsing strategies
    parsing_strategies = [
        # Standard JSON parsing
        lambda jt: json.loads(jt),
        
        # Fix unescaped quotes
        lambda jt: json.loads(jt.replace('\\"', '"').replace('""', '"')),
        
        # Fix single quotes used instead of double quotes
        lambda jt: json.loads(jt.replace("'", '"')),
        
        # Fix missing quotes around keys
        lambda jt: json.loads(re.sub(r'([{,])\s*(\w+):', r'\1"\2":', jt)),
        
        # Remove any JavaScript comments
        lambda jt: json.loads(re.sub(r'//.*?$|/\*.*?\*/', '', jt, flags=re.MULTILINE|re.DOTALL))
    ]
    
    for strategy in parsing_strategies:
        try:
            data = strategy(json_text)
            if isinstance(data, list):
                return [item.strip() for item in data if isinstance(item, str) and item.strip()]
        except Exception:
            continue
    
    return []

def try_extract_samples_with_regex(text: str) -> List[str]:
    """
    Use regex to extract items that look like they're part of an array.
    
    Args:
        text: Text that may contain items that look like array elements
        
    Returns:
        List of extracted strings or empty list if extraction fails
    """
    # Look for patterns that suggest array items:
    # 1. Items marked with numbers/bullets: "1. Item" or "- Item"
    numbered_items = re.findall(r'(?:^|\n)\s*(?:\d+\.|\-)\s*(.*?)(?=(?:\n\s*(?:\d+\.|\-))|$)', text, re.DOTALL)
    if numbered_items and len(numbered_items) > 1:
        return [item.strip() for item in numbered_items if item.strip()]
    
    # 2. Items in quotes within what looks like an array
    quoted_items = re.findall(r'["\'](.*?)["\'](?=\s*[,\]])', text)
    if quoted_items and len(quoted_items) > 1:
        return [item.strip() for item in quoted_items if item.strip()]
    
    # 3. Look for markdown code block that might contain JSON
    code_blocks = re.findall(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    for block in code_blocks:
        # Try to parse the code block as JSON
        samples = try_parse_json_array(block)
        if samples:
            return samples
    
    return []