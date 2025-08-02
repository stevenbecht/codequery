"""OpenAI embedding provider implementation."""

import logging
import time
import sys
from typing import List, Tuple, Dict, Any

import openai
from openai import OpenAI
import tiktoken

from .base import BaseEmbeddingProvider


def openai_with_retry(fn, *args, max_retries=5, base_wait=2, **kwargs):
    """
    Enhanced retry logic to handle various OpenAI errors.
    Updated for OpenAI API v1.0+
    """
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except (openai.RateLimitError, openai.APIStatusError) as e:
            sleep_time = base_wait * (2 ** attempt)
            logging.warning(f"[OpenAI API Error] Sleeping {sleep_time}s, attempt {attempt+1}/{max_retries}. Error: {e}")
            time.sleep(sleep_time)
        except openai.APITimeoutError:
            sleep_time = base_wait * (2 ** attempt)
            logging.warning(f"[OpenAI Timeout] Sleeping {sleep_time}s, attempt {attempt+1}/{max_retries}")
            time.sleep(sleep_time)
        except openai.APIConnectionError as e:
            logging.error(f"Could not connect to OpenAI API. Please check your internet connection. Error: {e}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected OpenAI API error: {e}")
            sys.exit(1)
    raise Exception("[Error] Max retries reached for OpenAI call.")


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI API-based embedding provider."""
    
    # Model dimensions mapping
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI provider."""
        super().__init__(config)
        self.model = config.get("openai_embed_model", "text-embedding-ada-002")
        self.client = OpenAI()
        
        # Validate API key
        if not openai.api_key:
            raise ValueError("OpenAI API key not set. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize tokenizer
        try:
            self._encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")
            logging.warning(f"Using default tokenizer for model {self.model}")
    
    def embed_batch(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Embed a batch of texts using OpenAI API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Tuple of (embeddings, total_tokens_used)
        """
        if not texts:
            return [], 0
        
        # Call OpenAI API with retry logic
        resp = openai_with_retry(
            self.client.embeddings.create,
            model=self.model,
            input=texts
        )
        
        # Extract embeddings and token usage
        embeddings = [item.embedding for item in resp.data]
        total_tokens = resp.usage.total_tokens
        
        return embeddings, total_tokens
    
    def get_token_count(self, text: str) -> int:
        """
        Count tokens using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self._encoding.encode(text))
    
    def get_vector_dim(self) -> int:
        """
        Get the dimension of OpenAI embedding vectors.
        
        Returns:
            Integer dimension of embedding vectors
        """
        return self.MODEL_DIMENSIONS.get(self.model, 1536)
    
    def get_model_name(self) -> str:
        """
        Get the OpenAI model name.
        
        Returns:
            Model name string
        """
        return self.model
    
    def get_provider_name(self) -> str:
        """
        Get the provider name.
        
        Returns:
            "openai"
        """
        return "openai"