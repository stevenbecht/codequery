import os
import openai
from openai import OpenAI
from dotenv import load_dotenv

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

    return {
        "openai_embed_model": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002"),
        "openai_chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
        "qdrant_host": os.getenv("QDRANT_HOST", "127.0.0.1"),
        "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
        "qdrant_collection": os.getenv("QDRANT_COLLECTION", "codebase_functions"),
        "max_chunk_tokens": int(os.getenv("CODEQUERY_MAX_CHUNK_TOKENS", "384"))
    }

