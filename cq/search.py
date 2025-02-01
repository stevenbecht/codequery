# BEGIN: cq/search.py
import logging
import openai
from qdrant_client import QdrantClient

from .embedding import count_tokens

def search_codebase_in_qdrant(
    query: str,
    collection_name: str,
    qdrant_client: QdrantClient,
    embed_model: str,
    top_k: int = 3,
    verbose: bool = False
):
    """Embed `query`, search Qdrant for top_k results."""
    query_emb = openai.Embedding.create(model=embed_model, input=query)["data"][0]["embedding"]
    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_emb,
        limit=top_k,
        with_payload=True,
        with_vectors=verbose
    )
    return response.points

def build_context_snippets(results):
    """Concatenate matched code snippets for LLM context (no token limit)."""
    segments = []
    for match in results:
        pl = match.payload
        snippet = (
            f"File: {pl['file_path']}, Function: {pl['function_name']} "
            f"(lines {pl['start_line']}-{pl['end_line']}):\n"
            f"{pl['code']}\n"
        )
        segments.append(snippet)
    return "\n---\n".join(segments)

def build_context_snippets_limited(
    results,
    max_context_tokens: int,
    model: str = "gpt-3.5-turbo",
    overhead_tokens: int = 200
):
    """
    Similar to build_context_snippets, but stops adding snippets
    once the total token count approaches max_context_tokens.
    """
    segments = []
    total = 0
    budget = max_context_tokens - overhead_tokens
    if budget < 0:
        budget = 0

    for match in results:
        pl = match.payload
        snippet_text = (
            f"File: {pl['file_path']}, Function: {pl['function_name']} "
            f"(lines {pl['start_line']}-{pl['end_line']}):\n"
            f"{pl['code']}\n"
        )
        snippet_tokens = count_tokens(snippet_text, model)

        if total + snippet_tokens > budget:
            break
        segments.append(snippet_text)
        total += snippet_tokens

    return "\n---\n".join(segments)

def chat_with_context(
    query: str,
    collection_name: str,
    qdrant_client: QdrantClient,
    embed_model: str,
    chat_model: str,
    top_k: int = 3,
    verbose: bool = False,
    max_context_tokens: int = None,
    reasoning_effort: str = "medium"
):
    """Search, build context, and send to OpenAI ChatCompletion."""
    import time
    from .embedding import count_tokens
    
    # Time the search operation
    search_start = time.time()
    query_tokens = count_tokens(query, embed_model)
    results = search_codebase_in_qdrant(query, collection_name, qdrant_client, embed_model, top_k, verbose)
    search_time = time.time() - search_start

    if verbose:
        logging.debug("=== Qdrant Search Results for Chat (Verbose) ===")
        for i, match in enumerate(results, start=1):
            logging.debug(f"--- Match #{i} ---")
            logging.debug(f"ID: {match.id}")
            logging.debug(f"Score: {match.score:.3f}")

            if match.vector is not None:
                logging.debug(f"Vector (first 8 dims): {match.vector[:8]} ...")

            pl = match.payload
            logging.debug(f"File: {pl['file_path']} | Function: {pl['function_name']}")
            logging.debug(f"(lines {pl['start_line']}-{pl['end_line']})\n")
            snippet_print = pl['code'][:200] + "..." if len(pl['code']) > 200 else pl['code']
            logging.debug(f"Snippet:\n{snippet_print}")
            logging.debug("-------------------------")

    if max_context_tokens is None or max_context_tokens <= 0:
        context_text = build_context_snippets(results)
    else:
        context_text = build_context_snippets_limited(
            results,
            max_context_tokens=max_context_tokens,
            model=chat_model
        )

    # Time the chat operation
    chat_start = time.time()
    
    if chat_model.startswith('o1-'):
        messages = [
            {"role": "user", "content": f"You are a helpful coding assistant. Here is the relevant code:\n{context_text}\n\nUser query: {query}"}
        ]
    elif chat_model == 'o3-mini':
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": f"Relevant code:\n{context_text}\n\nUser query: {query}"}
        ]
        # Add reasoning_effort parameter for o3-mini model
        resp = openai.ChatCompletion.create(
            model=chat_model,
            messages=messages,
            reasoning_effort=reasoning_effort
        )
        chat_time = time.time() - chat_start
        
        # Calculate token usage
        prompt_tokens = sum(count_tokens(m["content"], chat_model) for m in messages)
        completion_tokens = count_tokens(resp["choices"][0]["message"]["content"], chat_model)
        total_tokens = prompt_tokens + completion_tokens
        
        logging.info("\n=== Detailed Timing & Usage ===")
        logging.info(f"Search time: {search_time:.2f} seconds")
        logging.info(f"Search tokens: {query_tokens:,} tokens")
        logging.info(f"Chat time: {chat_time:.2f} seconds")
        logging.info(f"Chat tokens (prompt): {prompt_tokens:,} tokens")
        logging.info(f"Chat tokens (completion): {completion_tokens:,} tokens")
        logging.info(f"Chat tokens (total): {total_tokens:,} tokens")
        return resp["choices"][0]["message"]["content"]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": f"Relevant code:\n{context_text}\n\nUser query: {query}"}
        ]

    if verbose:
        logging.debug("=== OpenAI Chat Request (Verbose) ===")
        for idx, msg in enumerate(messages, 1):
            logging.debug(f"Message #{idx} [{msg['role']}]:")
            logging.debug(msg['content'])
            logging.debug("---")

    # Standard call for non-o3-mini models
    resp = openai.ChatCompletion.create(model=chat_model, messages=messages)
    chat_time = time.time() - chat_start
    
    # Calculate token usage
    prompt_tokens = sum(count_tokens(m["content"], chat_model) for m in messages)
    completion_tokens = count_tokens(resp["choices"][0]["message"]["content"], chat_model)
    total_tokens = prompt_tokens + completion_tokens
    
    logging.info("\n=== Detailed Timing & Usage ===")
    logging.info(f"Search time: {search_time:.2f} seconds")
    logging.info(f"Search tokens: {query_tokens:,} tokens")
    logging.info(f"Chat time: {chat_time:.2f} seconds")
    logging.info(f"Chat tokens (prompt): {prompt_tokens:,} tokens")
    logging.info(f"Chat tokens (completion): {completion_tokens:,} tokens")
    logging.info(f"Chat tokens (total): {total_tokens:,} tokens")
    
    return resp["choices"][0]["message"]["content"]
# END: cq/search.py
