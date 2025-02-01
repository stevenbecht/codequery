import sys
import logging

from requests.exceptions import ConnectionError as RequestsConnectionError
from qdrant_client import QdrantClient

def get_qdrant_client(host: str, port: int, verbose: bool = False) -> QdrantClient:
    """
    Attempt to create a QdrantClient.
    If there's a connection issue, log a more descriptive error and exit.
    """
    try:
        client = QdrantClient(host=host, port=port)
        return client
    except RequestsConnectionError as e:
        logging.error(f"Could not connect to Qdrant at {host}:{port}. Is Qdrant running?\nDetails: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected Qdrant connection error: {e}")
        sys.exit(1)
