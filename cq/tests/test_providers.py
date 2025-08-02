"""Basic tests for the embedding provider abstraction."""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from cq.providers import get_embedding_provider
from cq.providers.base import BaseEmbeddingProvider
from cq.providers.openai import OpenAIEmbeddingProvider


class TestProviderAbstraction:
    """Test the provider abstraction implementation."""
    
    def test_base_provider_interface(self):
        """Test that BaseEmbeddingProvider defines the required interface."""
        # BaseEmbeddingProvider is abstract, so we can't instantiate it directly
        assert hasattr(BaseEmbeddingProvider, 'embed_batch')
        assert hasattr(BaseEmbeddingProvider, 'get_token_count')
        assert hasattr(BaseEmbeddingProvider, 'get_vector_dim')
        assert hasattr(BaseEmbeddingProvider, 'get_model_name')
        assert hasattr(BaseEmbeddingProvider, 'get_provider_name')
        assert hasattr(BaseEmbeddingProvider, 'get_metadata')
        assert hasattr(BaseEmbeddingProvider, 'validate_compatibility')
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_openai_provider_creation(self):
        """Test creating an OpenAI provider."""
        config = {
            "embedding_provider": "openai",
            "openai_embed_model": "text-embedding-ada-002"
        }
        
        provider = get_embedding_provider(config)
        
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.get_provider_name() == "openai"
        assert provider.get_model_name() == "text-embedding-ada-002"
        assert provider.get_vector_dim() == 1536
    
    def test_provider_factory_with_local(self):
        """Test that the factory function handles local provider correctly."""
        config = {
            "embedding_provider": "local",
            "local_embed_model": "all-MiniLM-L6-v2"
        }
        
        # Since we don't have sentence-transformers installed in tests,
        # this should raise an ImportError
        with pytest.raises(ImportError) as excinfo:
            get_embedding_provider(config)
        assert "sentence-transformers" in str(excinfo.value)
    
    def test_provider_factory_unknown(self):
        """Test that unknown provider raises error."""
        config = {
            "embedding_provider": "unknown"
        }
        
        with pytest.raises(ValueError) as excinfo:
            get_embedding_provider(config)
        assert "Unknown embedding provider: unknown" in str(excinfo.value)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_openai_provider_metadata(self):
        """Test OpenAI provider metadata generation."""
        config = {
            "embedding_provider": "openai",
            "openai_embed_model": "text-embedding-ada-002"
        }
        
        provider = get_embedding_provider(config)
        metadata = provider.get_metadata()
        
        assert metadata["embedding_provider"] == "openai"
        assert metadata["embedding_model"] == "text-embedding-ada-002"
        assert metadata["vector_dimensions"] == 1536
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_compatibility_validation(self):
        """Test provider compatibility validation."""
        config = {
            "embedding_provider": "openai",
            "openai_embed_model": "text-embedding-ada-002"
        }
        
        provider = get_embedding_provider(config)
        
        # Compatible metadata
        compatible_metadata = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-ada-002",
            "vector_dimensions": 1536
        }
        
        # Should not raise
        provider.validate_compatibility(compatible_metadata)
        
        # Incompatible dimension
        incompatible_dim = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-ada-002",
            "vector_dimensions": 384
        }
        
        with pytest.raises(ValueError) as excinfo:
            provider.validate_compatibility(incompatible_dim)
        assert "Vector dimension mismatch" in str(excinfo.value)
        
        # Different provider
        different_provider = {
            "embedding_provider": "local",
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_dimensions": 384
        }
        
        with pytest.raises(ValueError) as excinfo:
            provider.validate_compatibility(different_provider)
        assert "Provider mismatch" in str(excinfo.value)


class TestProviderIntegration:
    """Test provider integration with mocked dependencies."""
    
    @patch('cq.providers.openai.OpenAI')
    @patch('cq.providers.openai.tiktoken.encoding_for_model')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_openai_embed_batch(self, mock_tiktoken, mock_openai_class):
        """Test OpenAI provider embed_batch method."""
        # Mock tiktoken encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3]
        mock_tiktoken.return_value = mock_encoder
        
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock embedding response
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_response.usage.total_tokens = 10
        
        mock_client.embeddings.create.return_value = mock_response
        
        # Create provider and test
        config = {
            "embedding_provider": "openai",
            "openai_embed_model": "text-embedding-ada-002"
        }
        
        provider = OpenAIEmbeddingProvider(config)
        embeddings, tokens = provider.embed_batch(["test text"])
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536
        assert tokens == 10
        
        # Verify the API was called correctly
        mock_client.embeddings.create.assert_called_once()
    
    @patch('cq.providers.openai.tiktoken.encoding_for_model')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_openai_token_counting(self, mock_tiktoken):
        """Test OpenAI provider token counting."""
        # Mock tiktoken encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
        mock_tiktoken.return_value = mock_encoder
        
        config = {
            "embedding_provider": "openai",
            "openai_embed_model": "text-embedding-ada-002"
        }
        
        provider = OpenAIEmbeddingProvider(config)
        token_count = provider.get_token_count("test text")
        
        assert token_count == 5
        mock_encoder.encode.assert_called_once_with("test text")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])