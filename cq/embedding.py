import os
import glob
import ast
import time
import hashlib
import logging
import sys

import openai
from openai import OpenAI
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from cq.config import load_config
from cq.providers import get_embedding_provider

# NEW: import pathspec for .gitignore handling
try:
    import pathspec
except ImportError:
    logging.warning("[GitIgnore] `pathspec` not installed. `.gitignore` patterns won't be applied.")
    pathspec = None

def count_tokens(text: str, model: str = "gpt-3.5-turbo", provider=None) -> int:
    """Return the token count of `text` using appropriate tokenizer."""
    if provider:
        # Use provider's token counting method
        return provider.get_token_count(text)
    else:
        # Fallback to tiktoken for backward compatibility
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))


def chunk_by_line_ranges(
    lines: list[str],
    file_path: str,
    func_name: str,
    start_line: int,
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 512,
    provider=None
):
    """
    Splits code by lines if a single block is too large for max_tokens.
    This is used for Python fallback (large functions) and for generic files.
    """
    current_chunk = []
    current_start_line = start_line

    for i, line_text in enumerate(lines, start=start_line):
        current_chunk.append(line_text)
        snippet = "".join(current_chunk)
        snippet_tokens = count_tokens(snippet, model, provider)

        if snippet_tokens > max_tokens:
            end_line = i - 1
            chunk_code = "".join(current_chunk[:-1])
            yield {
                "function_name": f"{func_name}_part",
                "start_line": current_start_line,
                "end_line": end_line,
                "code": chunk_code,
                "file_path": file_path,
            }
            current_chunk = [line_text]
            current_start_line = i

    if current_chunk:
        yield {
            "function_name": f"{func_name}_part",
            "start_line": current_start_line,
            "end_line": (start_line + len(lines) - 1),
            "code": "".join(current_chunk),
            "file_path": file_path,
        }

def chunk_file_python(file_path: str, model: str = "gpt-3.5-turbo", max_tokens: int = None, provider=None):
    """
    Parse a Python file into function-level chunks if possible,
    else fallback to line-based chunking for large functions.
    """
    if max_tokens is None:
        config = load_config()
        max_tokens = config["max_chunk_tokens"]
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        logging.error(f"Could not read {file_path}: {e}")
        return

    try:
        tree = ast.parse(source)
    except Exception as e:
        logging.error(f"Skipping {file_path}; AST parse error: {e}")
        return

    lines = source.splitlines(keepends=True)
    covered_lines = set()

    # Identify function-level chunks via AST
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1
            end_line = getattr(node, "end_lineno", None)
            if end_line is None:
                continue

            for ln in range(start_line, end_line):
                covered_lines.add(ln)

            code_snippet = "".join(lines[start_line:end_line])
            snippet_tokens = count_tokens(code_snippet, model, provider)

            if snippet_tokens > max_tokens:
                yield from chunk_by_line_ranges(
                    lines=lines[start_line:end_line],
                    file_path=file_path,
                    func_name=node.name,
                    start_line=start_line,
                    model=model,
                    max_tokens=max_tokens,
                    provider=provider
                )
            else:
                yield {
                    "function_name": node.name,
                    "start_line": start_line,
                    "end_line": end_line - 1,
                    "code": code_snippet,
                    "file_path": file_path
                }

    # Handle any top-level code not in functions
    top_level_segments = []
    for i, line_text in enumerate(lines):
        if i not in covered_lines:
            top_level_segments.append((i, line_text))

    if top_level_segments:
        combined_code = "".join(seg[1] for seg in top_level_segments)
        if count_tokens(combined_code, model, provider) > max_tokens:
            start_line = top_level_segments[0][0]
            yield from chunk_by_line_ranges(
                lines=[seg[1] for seg in top_level_segments],
                file_path=file_path,
                func_name="top_level",
                start_line=start_line,
                model=model,
                max_tokens=max_tokens,
                provider=provider
            )
        else:
            yield {
                "function_name": "top_level",
                "start_line": top_level_segments[0][0],
                "end_line": top_level_segments[-1][0],
                "code": combined_code,
                "file_path": file_path
            }

def chunk_file_generic(file_path: str, model: str = "gpt-3.5-turbo", max_tokens: int = None, provider=None):
    """
    Naive chunking for non-Python files (e.g., JS, TS, PHP).
    Reads the file as lines and chunks if needed based on max_tokens.
    """
    if max_tokens is None:
        config = load_config()
        max_tokens = config["max_chunk_tokens"]
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        logging.error(f"Could not read {file_path}: {e}")
        return

    # If entire file is within max_tokens, just yield once
    if count_tokens(source, model, provider) <= max_tokens:
        yield {
            "function_name": "entire_file",
            "start_line": 0,
            "end_line": len(source.splitlines()) - 1,
            "code": source,
            "file_path": file_path
        }
    else:
        # Otherwise, chunk by lines
        lines = source.splitlines(keepends=True)
        yield from chunk_by_line_ranges(
            lines=lines,
            file_path=file_path,
            func_name="entire_file",
            start_line=0,
            model=model,
            max_tokens=max_tokens,
            provider=provider
        )

def _load_gitignore_patterns(base_dir: str):
    """
    If pathspec is installed, parse the .gitignore in `base_dir`.
    Returns a compiled PathSpec, or None if no .gitignore found or pathspec missing.
    """
    if pathspec is None:
        return None  # pathspec not installed

    gitignore_path = os.path.join(base_dir, ".gitignore")
    if not os.path.isfile(gitignore_path):
        return None

    try:
        with open(gitignore_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, lines)
        return spec
    except Exception as e:
        logging.warning(f"[GitIgnore] Could not parse .gitignore: {e}")
        return None

def chunk_directory(
    directory: str,
    recursive: bool = False,
    model: str = "gpt-3.5-turbo",
    max_tokens: int = None,
    provider=None
):
    """
    Traverse the directory (optionally recursive) and yield code chunks
    from .py, .js, .ts, .php, .go files. Python uses AST-based chunking;
    other supported languages use naive line-based chunking.

    Now also honors `.gitignore` if present in `directory`.
    """
    if max_tokens is None:
        config = load_config()
        max_tokens = config["max_chunk_tokens"]
    
    gitignore_spec = _load_gitignore_patterns(directory)
    
    # Log once that we're using gitignore
    if gitignore_spec:
        logging.info(f"[ChunkDir] Using .gitignore filters from {directory}")

    if not recursive:
        # Non-recursive
        candidates = glob.glob(os.path.join(directory, "*.*"))
        for file_path in candidates:
            # If .gitignore pattern matches, skip
            if gitignore_spec:
                rel_path = os.path.relpath(file_path, directory)
                if gitignore_spec.match_file(rel_path):
                    logging.debug(f"[ChunkDir] Skipping ignored file: {rel_path}")
                    continue

            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".py":
                yield from chunk_file_python(file_path, model=model, max_tokens=max_tokens, provider=provider)
            elif ext in [".js", ".jsx", ".ts", ".tsx", ".php", ".go"]:
                yield from chunk_file_generic(file_path, model=model, max_tokens=max_tokens, provider=provider)
            else:
                logging.debug(f"[ChunkDir] Skipping unrecognized file type: {file_path}")
    else:
        # Recursive walk
        for root, dirs, files in os.walk(directory):
            # If .gitignore pattern matches the *directory*, remove it from `dirs` so we skip it entirely
            if gitignore_spec:
                dirs[:] = [
                    d for d in dirs
                    if not gitignore_spec.match_file(os.path.relpath(os.path.join(root, d), directory))
                ]

            for file_name in files:
                full_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(full_path, directory)

                if gitignore_spec and gitignore_spec.match_file(rel_path):
                    logging.debug(f"[ChunkDir] Skipping ignored file: {rel_path}")
                    continue

                ext = os.path.splitext(file_name)[1].lower()
                if ext == ".py":
                    yield from chunk_file_python(full_path, model=model, max_tokens=max_tokens, provider=provider)
                elif ext in [".js", ".jsx", ".ts", ".tsx", ".php", ".go"]:
                    yield from chunk_file_generic(full_path, model=model, max_tokens=max_tokens, provider=provider)
                else:
                    logging.debug(f"[ChunkDir] Skipping unrecognized file type: {full_path}")

def compute_snippet_hash(text: str) -> str:
    """Compute a hash (MD5) for the snippet text to detect changes."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def _store_collection_root_metadata(
    qdrant_client: QdrantClient,
    collection_name: str,
    root_dir: str,
    provider_metadata: dict = None
):
    """
    Upsert a special metadata point indicating the root_dir for this collection,
    so that we can later detect if the user is in a descendant folder.
    """
    # Store metadata with provider-specific vector dimensions
    vector_dim = 1536  # default for backward compatibility
    if provider_metadata:
        vector_dim = provider_metadata.get("vector_dimensions", 1536)
    
    dummy_vec = [0.0] * vector_dim
    payload = {
        "collection_meta": True,
        "root_dir": os.path.abspath(root_dir)
    }
    
    # Add provider metadata if available
    if provider_metadata:
        payload.update(provider_metadata)
    # We'll pick point ID = 0 for convenience
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": 0,
                    "vector": dummy_vec,
                    "payload": payload
                }
            ]
        )
        logging.debug(f"[Index] Stored collection root metadata for '{collection_name}' => {payload['root_dir']}")
    except Exception as e:
        logging.warning(f"[Index] Could not store root metadata in '{collection_name}': {e}")

def index_codebase_in_qdrant(
    directory: str,
    collection_name: str,
    qdrant_client: QdrantClient,
    embed_model: str,
    verbose: bool = False,
    recursive: bool = False,
    max_tokens: int = None,
    recreate: bool = False,
    provider=None
):
    """
    Index code from `directory` into Qdrant.
    By default, do incremental indexing if the collection already exists.
    If `recreate=True`, forcibly delete and re-create the collection first.

    Now also skips any files that match .gitignore in `directory`.
    Also stores file_mod_time and chunk_embed_time for each snippet.

    NEW: We also store a single "metadata" point with "root_dir" so that
    child folders can auto-detect the correct collection to use at query time.
    """
    config = load_config()
    if max_tokens is None:
        max_tokens = config["max_chunk_tokens"]
    
    # Get embedding provider if not provided
    if provider is None:
        provider = get_embedding_provider(config)
    
    # Get provider metadata
    provider_metadata = provider.get_metadata()
    vector_dim = provider.get_vector_dim()
    
    # Log provider information
    logging.info(f"[Index] Using {provider.get_provider_name()} provider with model {provider.get_model_name()}")
    
    # If user wants a fresh index:
    if recreate:
        if qdrant_client.collection_exists(collection_name):
            logging.info(f"[Index] Re-creating collection '{collection_name}'")
            qdrant_client.delete_collection(collection_name)
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        known_hashes = set()
    else:
        # Incremental approach
        if not qdrant_client.collection_exists(collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
            )
            known_hashes = set()
        else:
            known_hashes = set()
            limit = 100
            offset = None

            while True:
                points_batch, next_offset = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                for point in points_batch:
                    if point.payload and "snippet_hash" in point.payload:
                        known_hashes.add(point.payload["snippet_hash"])
                if next_offset is None:
                    break
                offset = next_offset

            if verbose:
                logging.debug(f"[Index] Incremental mode: found {len(known_hashes)} existing snippet hashes.")

    # Store or refresh the root dir metadata with provider info
    _store_collection_root_metadata(qdrant_client, collection_name, directory, provider_metadata)

    all_chunks = list(
        chunk_directory(
            directory=directory,
            recursive=recursive,
            model=embed_model,
            max_tokens=max_tokens,
            provider=provider
        )
    )
    if not all_chunks:
        logging.info(f"No supported code files found in '{directory}' (or all ignored).")
        return

    # Find the next point_id to assign:
    if recreate:
        point_id = 1  # because we used 0 for the metadata
    else:
        # get max existing ID
        max_id = 0
        offset = None
        while True:
            points_batch, next_offset = qdrant_client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )
            for point in points_batch:
                if point.id > max_id:
                    max_id = point.id
            if next_offset is None:
                break
            offset = next_offset
        point_id = max_id + 1
        if verbose:
            logging.debug(f"[Index] Next point_id will start at {point_id}")

    chunks_to_embed = []
    for ch in all_chunks:
        snippet_hash = compute_snippet_hash(ch["code"])
        ch["snippet_hash"] = snippet_hash
        # NEW: store disk mod time and embed time
        try:
            ch["file_mod_time"] = os.path.getmtime(ch["file_path"])
        except Exception:
            ch["file_mod_time"] = 0.0
        ch["chunk_embed_time"] = time.time()

        if snippet_hash not in known_hashes:
            chunks_to_embed.append(ch)
        else:
            if verbose:
                logging.debug(f"[Index] Skipping unchanged snippet: {ch['file_path']} "
                              f"({ch['start_line']}-{ch['end_line']})")

    if not chunks_to_embed:
        logging.info("[Index] All snippets are up-to-date. No new embeddings.")
        return

    if verbose:
        logging.debug(f"[Index] Found {len(chunks_to_embed)} new/changed snippets to embed.")

    # Use configured batch size from provider
    BATCH_SIZE = config.get("embed_batch_size", 100)
    total_tokens = 0

    # Split into batches so we don't exceed OpenAI's request size limit
    for start_idx in range(0, len(chunks_to_embed), BATCH_SIZE):
        batch = chunks_to_embed[start_idx:start_idx + BATCH_SIZE]
        texts = [b["code"] for b in batch]
        vectors, used_tokens = provider.embed_batch(texts)
        total_tokens += used_tokens

        points_to_upsert = []
        for i, vec in enumerate(vectors):
            chunk_data = batch[i]
            snippet_token_count = count_tokens(chunk_data["code"], embed_model, provider)
            payload = {
                "file_path": chunk_data["file_path"],
                "function_name": chunk_data["function_name"],
                "start_line": chunk_data["start_line"],
                "end_line": chunk_data["end_line"],
                "code": chunk_data["code"],
                "chunk_tokens": snippet_token_count,
                "snippet_hash": chunk_data["snippet_hash"],
                # NEW: store mod time + embed time
                "file_mod_time": chunk_data["file_mod_time"],
                "chunk_embed_time": chunk_data["chunk_embed_time"],
            }
            points_to_upsert.append({
                "id": point_id,
                "vector": vec,
                "payload": payload
            })
            point_id += 1

        qdrant_client.upsert(collection_name=collection_name, points=points_to_upsert)

        if verbose:
            logging.debug(
                f"[Index] Upserted {len(points_to_upsert)} points. Running total tokens: {total_tokens}"
            )

    logging.info(f"[Index] Done embedding. Estimated total billed tokens = {total_tokens}")
