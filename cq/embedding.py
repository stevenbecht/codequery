import os
import glob
import ast
import time
import hashlib
import logging

import openai
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Return the token count of `text` using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def openai_with_retry(fn, *args, max_retries=5, base_wait=2, **kwargs):
    """
    Simple retry logic to handle RateLimitError or Timeout from OpenAI.
    """
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except openai.error.RateLimitError:
            sleep_time = base_wait * (2 ** attempt)
            logging.warning(f"[RateLimitError] Sleeping {sleep_time}s, attempt {attempt+1}/{max_retries}")
            time.sleep(sleep_time)
        except openai.error.Timeout:
            sleep_time = base_wait * (2 ** attempt)
            logging.warning(f"[Timeout] Sleeping {sleep_time}s, attempt {attempt+1}/{max_retries}")
            time.sleep(sleep_time)
    raise Exception("[Error] Max retries reached for OpenAI call.")

def chunk_by_line_ranges(
    lines: list[str],
    file_path: str,
    func_name: str,
    start_line: int,
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 512
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
        snippet_tokens = count_tokens(snippet, model)

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

def chunk_file_python(file_path: str, model: str = "gpt-3.5-turbo", max_tokens: int = 1500):
    """
    Parse a Python file into function-level chunks if possible,
    else fallback to line-based chunking for large functions.
    """
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
            snippet_tokens = count_tokens(code_snippet, model)

            if snippet_tokens > max_tokens:
                yield from chunk_by_line_ranges(
                    lines=lines[start_line:end_line],
                    file_path=file_path,
                    func_name=node.name,
                    start_line=start_line,
                    model=model,
                    max_tokens=max_tokens
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
        if count_tokens(combined_code, model) > max_tokens:
            start_line = top_level_segments[0][0]
            yield from chunk_by_line_ranges(
                lines=[seg[1] for seg in top_level_segments],
                file_path=file_path,
                func_name="top_level",
                start_line=start_line,
                model=model,
                max_tokens=max_tokens
            )
        else:
            yield {
                "function_name": "top_level",
                "start_line": top_level_segments[0][0],
                "end_line": top_level_segments[-1][0],
                "code": combined_code,
                "file_path": file_path
            }

def chunk_file_generic(file_path: str, model: str = "gpt-3.5-turbo", max_tokens: int = 1500):
    """
    Naive chunking for non-Python files (e.g., JS, TS, PHP).
    Reads the file as lines and chunks if needed based on max_tokens.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        logging.error(f"Could not read {file_path}: {e}")
        return

    # If entire file is within max_tokens, just yield once
    if count_tokens(source, model) <= max_tokens:
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
            max_tokens=max_tokens
        )

def chunk_directory(
    directory: str,
    recursive: bool = False,
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 1500
):
    """
    Traverse the directory (optionally recursive) and yield code chunks
    from .py, .js, .ts, .php files. Python uses AST-based chunking;
    other supported languages use naive line-based chunking.
    """
    if not recursive:
        candidates = glob.glob(os.path.join(directory, "*.*"))
        for file_path in candidates:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".py":
                yield from chunk_file_python(file_path, model=model, max_tokens=max_tokens)
            elif ext in [".js", ".jsx", ".ts", ".tsx", ".php"]:
                yield from chunk_file_generic(file_path, model=model, max_tokens=max_tokens)
            else:
                logging.info(f"[ChunkDir] Skipping unrecognized file type: {file_path}")
    else:
        for root, dirs, files in os.walk(directory):
            for file_name in files:
                ext = os.path.splitext(file_name)[1].lower()
                full_path = os.path.join(root, file_name)
                if ext == ".py":
                    yield from chunk_file_python(full_path, model=model, max_tokens=max_tokens)
                elif ext in [".js", ".jsx", ".ts", ".tsx", ".php"]:
                    yield from chunk_file_generic(full_path, model=model, max_tokens=max_tokens)
                else:
                    logging.info(f"[ChunkDir] Skipping unrecognized file type: {full_path}")

def compute_snippet_hash(text: str) -> str:
    """Compute a hash (MD5) for the snippet text to detect changes."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def index_codebase_in_qdrant(
    directory: str,
    collection_name: str,
    qdrant_client: QdrantClient,
    embed_model: str,
    verbose: bool = False,
    recursive: bool = False,
    max_tokens: int = 1500,
    recreate: bool = False  # <-- new param
):
    """
    Index code from `directory` into Qdrant.
    By default, do incremental indexing if the collection already exists.
    If `recreate=True`, forcibly delete and re-create the collection first.
    """
    # If user wants a fresh index:
    if recreate:
        if qdrant_client.collection_exists(collection_name):
            logging.info(f"[Index] Re-creating collection '{collection_name}'")
            qdrant_client.delete_collection(collection_name)
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        known_hashes = set()
    else:
        # Incremental approach
        if not qdrant_client.collection_exists(collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
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

    all_chunks = list(
        chunk_directory(
            directory=directory,
            recursive=recursive,
            model=embed_model,
            max_tokens=max_tokens
        )
    )
    if not all_chunks:
        logging.info(f"No supported code files found in '{directory}'.")
        return

    # Find the next point_id to assign:
    if recreate:
        point_id = 0
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
        if snippet_hash not in known_hashes:
            ch["snippet_hash"] = snippet_hash
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

    BATCH_SIZE = 100
    total_tokens = 0

    def embed_batch(texts: list[str], model: str):
        resp = openai_with_retry(
            openai.Embedding.create,
            model=model,
            input=texts
        )
        used_tokens = resp["usage"]["total_tokens"]
        vectors = [item["embedding"] for item in resp["data"]]
        return vectors, used_tokens

    # Split into batches so we don’t exceed OpenAI’s request size limit
    for start_idx in range(0, len(chunks_to_embed), BATCH_SIZE):
        batch = chunks_to_embed[start_idx:start_idx + BATCH_SIZE]
        texts = [b["code"] for b in batch]
        vectors, used_tokens = embed_batch(texts, embed_model)
        total_tokens += used_tokens

        points_to_upsert = []
        for i, vec in enumerate(vectors):
            chunk_data = batch[i]
            snippet_token_count = count_tokens(chunk_data["code"], embed_model)
            payload = {
                "file_path": chunk_data["file_path"],
                "function_name": chunk_data["function_name"],
                "start_line": chunk_data["start_line"],
                "end_line": chunk_data["end_line"],
                "code": chunk_data["code"],
                "chunk_tokens": snippet_token_count,
                "snippet_hash": chunk_data["snippet_hash"]
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
