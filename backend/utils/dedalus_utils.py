"""
Utility functions for Dedalus Labs API queries.
"""
import os
import logging
from openai import OpenAI
from typing import Optional

logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL = os.getenv("DEDALUS_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))

# Client will be initialized lazily
_client = None


def _get_client():
    """
    Get or create the OpenAI client.
    Initializes on first use to avoid import-time errors.
    """
    global _client
    if _client is None:
        DEDALUS_API_KEY = os.getenv("DEDALUS_API_KEY")
        DEDALUS_BASE_URL = os.getenv("DEDALUS_BASE_URL", "https://api.dedaluslabs.ai/v1")
        
        if not DEDALUS_API_KEY:
            raise ValueError("DEDALUS_API_KEY environment variable is required.")
        
        _client = OpenAI(
            api_key=DEDALUS_API_KEY,
            base_url=DEDALUS_BASE_URL
        )
    return _client


def run_ai_query(query: str, model: Optional[str] = None, json_mode: bool = False) -> str:
    """
    Run an AI query using Dedalus Labs API.
    
    Args:
        query: The query string to send to the AI model
        model: Optional model name, defaults to DEFAULT_MODEL
        json_mode: If True, request JSON response format (if supported by model)
        
    Returns:
        The response text from the AI model
        
    Raises:
        Exception: If the API call fails
    """
    if model is None:
        model = DEFAULT_MODEL
    
    client = _get_client()
    
    try:
        # Build request parameters
        request_params = {
            "model": model,
            "messages": [
                {"role": "user", "content": query}
            ],
            "temperature": 0.7,
            "max_tokens": 2000  # Increased for more complex responses
        }
        
        # Try to use JSON mode if requested and supported
        if json_mode:
            try:
                request_params["response_format"] = {"type": "json_object"}
            except Exception:
                # If response_format is not supported, continue without it
                logger.debug("JSON response format not supported, continuing without it")
        
        response = client.chat.completions.create(**request_params)
        
        content = response.choices[0].message.content
        return content.strip()
        
    except Exception as e:
        logger.error(f"Error running AI query: {str(e)}")
        raise

