# cq/shared.py
"""
Shared utilities for CodeQuery commands.
Consolidates common functionality to avoid code duplication.
"""

import os
import sys
import logging
from typing import Optional

from requests.exceptions import ConnectionError as RequestsConnectionError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

# Re-export count_tokens from embedding module for convenience
from cq.embedding import count_tokens

def get_qdrant_client(host: str, port: int, verbose: bool = False) -> QdrantClient:
    """
    Attempt to create a QdrantClient.
    If there's a connection issue, log a more descriptive error and exit.
    """
    try:
        client = QdrantClient(host=host, port=port)
        # Test the connection
        client.get_collections()  # This will verify we can actually connect
        return client
    except RequestsConnectionError as e:
        logging.error(f"Could not connect to Qdrant at {host}:{port}. Is Qdrant running?\nTry running: ./run_qdrant.sh\nDetails: {e}")
        sys.exit(1)
    except ResponseHandlingException as e:
        logging.error(f"Failed to communicate with Qdrant at {host}:{port}. Is Qdrant running?\nTry running: ./run_qdrant.sh\nDetails: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected Qdrant connection error: {e}")
        sys.exit(1)

def find_collection_for_current_dir(client: QdrantClient, current_dir: str) -> Optional[str]:
    """
    Auto-detect which Qdrant collection corresponds to 'current_dir'
    by checking the special 'collection_meta' record that stores 'root_dir'.
    If multiple matches, pick the one with the longest root_dir (most specific).
    Return the matching collection name, or None if none found.
    """
    try:
        collections_info = client.get_collections()
        all_collections = [c.name for c in collections_info.collections]
        best_match = None
        best_len = 0

        for coll_name in all_collections:
            # We'll scroll to find that metadata record
            try:
                points_batch, _ = client.scroll(
                    collection_name=coll_name,
                    limit=1000,  # should be enough to find id=0 if it exists
                    with_payload=True,
                    with_vectors=False
                )
                for p in points_batch:
                    pl = p.payload
                    if not pl:
                        continue
                    if pl.get("collection_meta") and "root_dir" in pl:
                        root_dir = os.path.abspath(pl["root_dir"])
                        cur_dir_abs = os.path.abspath(current_dir)
                        common = os.path.commonpath([root_dir, cur_dir_abs])
                        if common == root_dir:
                            # current_dir is inside root_dir
                            rlen = len(root_dir)
                            if rlen > best_len:
                                best_len = rlen
                                best_match = coll_name
            except:
                pass

        return best_match
    except Exception as e:
        logging.debug(f"Error in find_collection_for_current_dir: {e}")
        return None

def ensure_collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """
    Check if a collection exists in Qdrant.
    Returns True if exists, False otherwise.
    """
    return client.collection_exists(collection_name)

def log_progress(message: str, end: str = "\n"):
    """
    Standardized progress logging that respects the 'end' parameter.
    Useful for progress messages that update on the same line.
    """
    logging.info(message, extra={"end": end})