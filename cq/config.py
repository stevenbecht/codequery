import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

def load_config():
    """
    Loads environment variables from .env, sets OpenAI key,
    and returns relevant config parameters.
    """
    load_dotenv()

    # Set API key for backward compatibility with older code
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    
    # The OpenAI client will automatically use OPENAI_API_KEY environment variable
    # So this is optional, but shown for clarity
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    
    # Determine default cache directory
    cache_home = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    default_cache_dir = os.path.join(cache_home, "codequery")
    
    # Determine default batch size based on provider
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    default_batch_size = "100" if embedding_provider == "openai" else "32"

    return {
        # Existing OpenAI settings
        "openai_embed_model": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002"),
        "openai_chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
        
        # New embedding provider settings
        "embedding_provider": embedding_provider,
        "local_embed_model": os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2"),
        "local_embed_cache_dir": os.getenv("LOCAL_EMBED_CACHE_DIR", default_cache_dir),
        "embed_batch_size": int(os.getenv("EMBED_BATCH_SIZE", default_batch_size)),
        "embed_device": os.getenv("EMBED_DEVICE", "auto"),
        
        # Qdrant settings
        "qdrant_host": os.getenv("QDRANT_HOST", "127.0.0.1"),
        "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
        "qdrant_collection": os.getenv("QDRANT_COLLECTION", "codebase_functions"),
        
        # Other settings
        "max_chunk_tokens": int(os.getenv("CODEQUERY_MAX_CHUNK_TOKENS", "384"))
    }

