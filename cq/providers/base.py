"""Base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Tuple of (embeddings, total_tokens_used)
            - embeddings: List of embedding vectors (each is a list of floats)
            - total_tokens_used: Total number of tokens processed
        """
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Count tokens in a text string using the provider's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass
    
    @abstractmethod
    def get_vector_dim(self) -> int:
        """
        Get the dimension of embedding vectors produced by this provider.
        
        Returns:
            Integer dimension of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the embedding model.
        
        Returns:
            Model name string
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            Provider name string (e.g., "openai", "local")
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this provider for storage in collections.
        
        Returns:
            Dictionary with provider metadata
        """
        return {
            "embedding_provider": self.get_provider_name(),
            "embedding_model": self.get_model_name(),
            "vector_dimensions": self.get_vector_dim()
        }
    
    def validate_compatibility(self, collection_metadata: Dict[str, Any]) -> None:
        """
        Validate that this provider is compatible with a collection.
        
        Args:
            collection_metadata: Metadata from the collection
            
        Raises:
            ValueError: If provider is incompatible with the collection
        """
        expected_dim = self.get_vector_dim()
        actual_dim = collection_metadata.get("vector_dimensions")
        
        if actual_dim and actual_dim != expected_dim:
            raise ValueError(
                f"Vector dimension mismatch: collection has {actual_dim} dimensions "
                f"but {self.get_provider_name()} provider with model {self.get_model_name()} "
                f"produces {expected_dim} dimensions"
            )
        
        # Warn if using different provider/model
        collection_provider = collection_metadata.get("embedding_provider")
        collection_model = collection_metadata.get("embedding_model")
        
        if collection_provider and collection_provider != self.get_provider_name():
            raise ValueError(
                f"Provider mismatch: collection was created with {collection_provider} "
                f"but trying to use {self.get_provider_name()}"
            )
        
        if collection_model and collection_model != self.get_model_name():
            raise ValueError(
                f"Model mismatch: collection was created with {collection_model} "
                f"but trying to use {self.get_model_name()}"
            )