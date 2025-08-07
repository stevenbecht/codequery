"""Local embedding provider using sentence-transformers."""

import os
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path

from .base import BaseEmbeddingProvider


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding provider using sentence-transformers models."""
    
    # Model dimensions mapping for common models
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "paraphrase-MiniLM-L3-v2": 384,  # Tiny model for testing
        "all-MiniLM-L12-v2": 384,
        "all-distilroberta-v1": 768,
        "multi-qa-MiniLM-L6-cos-v1": 384,
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize local embedding provider."""
        super().__init__(config)
        
        # Check if sentence-transformers is available
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError(
                "Local embedding provider requires sentence-transformers and torch. "
                "Install with: pip install codequery[local]"
            )
        
        self.model_name = config.get("local_embed_model", "all-MiniLM-L6-v2")
        self.cache_dir = config.get("local_embed_cache_dir", "~/.cache/codequery")
        self.cache_dir = os.path.expanduser(self.cache_dir)
        self.batch_size = config.get("embed_batch_size", 32)
        
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Device selection
        device_config = config.get("embed_device", "auto").lower()
        if device_config == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_config in ["cuda", "cpu", "mps"]:
            self.device = device_config
        else:
            logging.warning(f"Unknown device '{device_config}', defaulting to CPU")
            self.device = "cpu"
        
        logging.info(f"Initializing local embedding model '{self.model_name}' on {self.device}")
        
        # Load the model
        try:
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )
            
            # Get the actual dimension from the model
            self._vector_dim = self.model.get_sentence_embedding_dimension()
            
            # Verify against expected dimension if known
            expected_dim = self.MODEL_DIMENSIONS.get(self.model_name)
            if expected_dim and expected_dim != self._vector_dim:
                logging.warning(
                    f"Model {self.model_name} has dimension {self._vector_dim}, "
                    f"expected {expected_dim}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.model_name}': {e}")
        
        # Cache the tokenizer for token counting
        self._tokenizer = self.model.tokenizer
    
    def embed_batch(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Embed a batch of texts using the local model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Tuple of (embeddings, total_tokens_used)
        """
        if not texts:
            return [], 0
        
        logging.debug(f"Processing {len(texts)} texts for embedding")
        
        # Safety check: warn about oversized texts and truncate if needed
        # Use 500 to be safe with special tokens
        max_length = 500
        checked_texts = []
        total_tokens = 0
        truncated_count = 0
        
        for i, text in enumerate(texts):
            token_count = self.get_token_count(text)
            total_tokens += token_count
            
            if token_count > max_length:
                truncated_count += 1
                logging.debug(f"Text {i+1}/{len(texts)} has {token_count} tokens, truncating to {max_length}")
                # Truncate using the tokenizer's built-in truncation
                tokens = self._tokenizer(
                    text, 
                    max_length=max_length, 
                    truncation=True, 
                    add_special_tokens=True,
                    return_tensors=None
                )
                truncated_text = self._tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)
                checked_texts.append(truncated_text)
            else:
                checked_texts.append(text)
        
        if truncated_count > 0:
            logging.info(f"Truncated {truncated_count} oversized text chunks to fit model limit")
        
        # Generate embeddings with checked texts
        embeddings = self.model.encode(
            checked_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # SBERT models are typically normalized
            convert_to_numpy=True  # Convert to numpy for consistent handling
        )
        
        # Always convert to list of lists for Qdrant compatibility
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()
        else:
            # Handle single embedding case
            embeddings = [embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding) for embedding in embeddings]
        
        return embeddings, total_tokens
    
    def get_token_count(self, text: str) -> int:
        """
        Count tokens using the model's tokenizer.
        Includes special tokens for accurate count.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        # Use fast tokenizer backend if available (no warning)
        if getattr(self._tokenizer, "is_fast", False):
            # Fast tokenizers have a backend that doesn't emit warnings
            encoded = self._tokenizer.backend_tokenizer.encode(text, add_special_tokens=True)
            return len(encoded.ids)
        
        # Fallback for slow tokenizers: silence Transformers logging for this call
        from transformers.utils import logging as hf_logging
        with hf_logging.verbosity_error():
            tokens = self._tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )
        return len(tokens)
    
    def get_vector_dim(self) -> int:
        """
        Get the dimension of embedding vectors.
        
        Returns:
            Integer dimension of embedding vectors
        """
        return self._vector_dim
    
    def get_model_name(self) -> str:
        """
        Get the model name.
        
        Returns:
            Model name string
        """
        return self.model_name
    
    def get_provider_name(self) -> str:
        """
        Get the provider name.
        
        Returns:
            "local"
        """
        return "local"
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this provider for storage in collections.
        
        Returns:
            Dictionary with provider metadata
        """
        metadata = super().get_metadata()
        metadata["device"] = self.device
        metadata["cache_dir"] = self.cache_dir
        return metadata
    
    @staticmethod
    def list_available_models(cache_dir: str = None) -> List[str]:
        """
        List models available in the cache directory.
        
        Args:
            cache_dir: Cache directory to check (default: ~/.cache/codequery)
            
        Returns:
            List of available model names
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/codequery")
        
        models = []
        if os.path.exists(cache_dir):
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                    models.append(item)
        
        return sorted(models)
    
    @staticmethod
    def download_model(model_name: str, cache_dir: str = None, from_path: str = None) -> None:
        """
        Download a model or copy from local path.
        
        Args:
            model_name: Name of the model to download
            cache_dir: Directory to cache the model
            from_path: Local path to copy model from (for offline installation)
        """
        try:
            from sentence_transformers import SentenceTransformer
            import shutil
        except ImportError:
            raise ImportError(
                "sentence-transformers is required to download models. "
                "Install with: pip install codequery[local]"
            )
        
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/codequery")
        
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        if from_path:
            # Copy from local path for offline installation
            if not os.path.exists(from_path):
                raise ValueError(f"Source path does not exist: {from_path}")
            
            target_path = os.path.join(cache_dir, model_name)
            logging.info(f"Copying model from {from_path} to {target_path}")
            
            if os.path.exists(target_path):
                logging.warning(f"Model already exists at {target_path}, skipping")
                return
            
            shutil.copytree(from_path, target_path)
            logging.info("Model copied successfully")
        else:
            # Download from HuggingFace
            logging.info(f"Downloading model '{model_name}' to {cache_dir}")
            model = SentenceTransformer(model_name, cache_folder=cache_dir)
            logging.info(f"Model '{model_name}' downloaded successfully")