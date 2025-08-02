"""Embedding provider implementations for CodeQuery."""

from .base import BaseEmbeddingProvider
from .openai import OpenAIEmbeddingProvider

__all__ = ["BaseEmbeddingProvider", "OpenAIEmbeddingProvider"]

def get_embedding_provider(config):
    """Factory function to get the appropriate embedding provider based on config."""
    provider_type = config.get("embedding_provider", "openai").lower()
    
    if provider_type == "openai":
        return OpenAIEmbeddingProvider(config)
    elif provider_type == "local":
        # Lazy import to avoid requiring torch/sentence-transformers for OpenAI-only users
        from .local import LocalEmbeddingProvider
        return LocalEmbeddingProvider(config)
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")