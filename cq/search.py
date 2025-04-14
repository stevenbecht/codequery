import logging
import openai
from openai import OpenAI
from qdrant_client import QdrantClient
import time
import sys
import threading

from .embedding import count_tokens

def get_model_token_limit(model_name):
    """
    Get the token limit for various OpenAI models.
    Returns the token limit for the given model or a default.
    """
    model_token_limits = {
        # Embedding models
        "text-embedding-ada-002": 8191,
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
        
        # Chat models
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4-1106-preview": 128000,
        "gpt-4-0125-preview": 128000,
        "gpt-4-1106-vision-preview": 128000,
        "gpt-4-vision-preview": 128000,
        "gpt-4.1": 128000
    }
    return model_token_limits.get(model_name, 8000)  # Default to 8000 if model is unknown

def search_codebase_in_qdrant(
    query: str,
    collection_name: str,
    qdrant_client: QdrantClient,
    embed_model: str,
    top_k: int = 3,
    verbose: bool = False
):
    """Embed `query`, search Qdrant for top_k results."""
    # Get query embedding and count tokens
    query_tokens = count_tokens(query, embed_model)
    
    # Check model token limits
    token_limit = get_model_token_limit(embed_model)
    
    if query_tokens > token_limit:
        raise ValueError(
            f"Query exceeds the token limit for model {embed_model}. "
            f"Token count: {query_tokens}, limit: {token_limit}. "
            f"Try reducing the amount of code or using a smaller context."
        )
        
    client = OpenAI()
    query_emb = client.embeddings.create(model=embed_model, input=query).data[0].embedding
    
    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_emb,
        limit=top_k,
        with_payload=True,
        with_vectors=verbose
    )
    
    # Calculate total tokens in matched snippets
    total_snippet_tokens = sum(
        count_tokens(point.payload['code'], embed_model) 
        for point in response.points
    )
    
    return {
        'points': response.points,
        'query_tokens': query_tokens,
        'snippet_tokens': total_snippet_tokens,
        'total_tokens': query_tokens + total_snippet_tokens
    }

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

    truncated = False
    total_snippets = len(results)
    included_snippets = 0
    
    # First, calculate tokens for all snippets
    all_snippets_tokens = 0
    for match in results:
        pl = match.payload
        snippet_text = (
            f"File: {pl['file_path']}, Function: {pl['function_name']} "
            f"(lines {pl['start_line']}-{pl['end_line']}):\n"
            f"{pl['code']}\n"
        )
        snippet_tokens = count_tokens(snippet_text, model)
        all_snippets_tokens += snippet_tokens
    
    # Now build the context that fits within the budget
    for match in results:
        pl = match.payload
        snippet_text = (
            f"File: {pl['file_path']}, Function: {pl['function_name']} "
            f"(lines {pl['start_line']}-{pl['end_line']}):\n"
            f"{pl['code']}\n"
        )
        snippet_tokens = count_tokens(snippet_text, model)

        if total + snippet_tokens > budget:
            truncated = True
            break
        segments.append(snippet_text)
        total += snippet_tokens
        included_snippets += 1

    if truncated:
        logging.warning(f"[Context] Only included {included_snippets}/{total_snippets} snippets due to token limit.")
        file_paths = set(match.payload['file_path'] for match in results)
        logging.warning(f"[Context] Found {len(file_paths)} relevant files, but only showing content from {included_snippets} snippets.")
        logging.warning(f"[Context] Total tokens needed: {all_snippets_tokens}, token window: {max_context_tokens}")
        logging.warning("[Context] Use -w/--max-window with a higher value to include more context (e.g., -w 8000)")

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
    # Time the search operation
    search_start = time.time()
    query_tokens = count_tokens(query, embed_model)
    
    try:
        results = search_codebase_in_qdrant(query, collection_name, qdrant_client, embed_model, top_k, verbose)
    except ValueError as e:
        if "exceeds the token limit" in str(e):
            logging.error(f"Token limit exceeded: {e}")
            logging.error(f"Your code context is too large ({query_tokens} tokens).")
            logging.error("Suggestions:")
            logging.error("1. Use -n/--no-context flag to chat without code context")
            logging.error("2. Use a smaller directory with 'cq dump <smaller_dir>'")
            logging.error("3. Use fewer files by being more specific with your dump paths")
            return None
        else:
            # Re-raise other ValueError exceptions
            raise
    except Exception as e:
        logging.error(f"Error during search: {e}")
        return None
        
    search_time = time.time() - search_start
    
    # Extract the list of files used in the context
    context_files = []
    for match in results['points']:
        file_path = match.payload['file_path']
        if file_path not in context_files:
            context_files.append(file_path)
            
    # Display the files found in search before building context
    if context_files:
        logging.info("=== Files Found in Search ===")
        for file in context_files:
            logging.info(f"- {file}")

    if verbose:
        logging.debug("=== Qdrant Search Results for Chat (Verbose) ===")
        for i, match in enumerate(results['points'], start=1):
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
        context_text = build_context_snippets(results['points'])
    else:
        context_text = build_context_snippets_limited(
            results['points'],
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
        
        # Calculate and show input tokens
        prompt_tokens = sum(count_tokens(m["content"], chat_model) for m in messages)
        
        # Check if we're going to exceed the model's token limit
        token_limit = get_model_token_limit(chat_model)
        if prompt_tokens > token_limit:
            logging.error(f"Chat prompt exceeds the token limit for model {chat_model}.")
            logging.error(f"Token count: {prompt_tokens:,}, limit: {token_limit:,}")
            logging.error("\nSuggestions:")
            logging.error("1. Use -n/--no-context flag to chat without code context")
            logging.error("2. Use a smaller directory with 'cq dump <smaller_dir>'")
            logging.error("3. Use a model with larger context (e.g. --model 'openai:gpt-4-turbo')")
            logging.error("4. Use -w/--max-window to limit context tokens (e.g. -w 4000)")
            return None
        
        # Show context info AFTER files list but BEFORE API call
        logging.info(f"Input tokens: {prompt_tokens:,}")
        logging.info("Sending request to OpenAI API...")
        
        # Setup progress indicator thread
        stop_progress = threading.Event()
        request_start = time.time()
        
        # Thread function to show progress while waiting
        def show_progress():
            while not stop_progress.is_set():
                elapsed = time.time() - request_start
                sys.stdout.write(f"\rWaiting for response... [{elapsed:.1f}s]")
                sys.stdout.flush()
                time.sleep(1.0)
        
        # Start progress thread
        progress_thread = threading.Thread(target=show_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        # Make the API call
        try:
            # Add reasoning_effort parameter for o3-mini model
            client = OpenAI()
            resp = client.chat.completions.create(
                model=chat_model,
                messages=messages,
                reasoning_effort=reasoning_effort
            )
            
            # Stop the progress indicator
            stop_progress.set()
            progress_thread.join()
            
            # Add a newline after our counter
            sys.stdout.write("\n")
            sys.stdout.flush()
            
        except Exception as e:
            # Stop the progress indicator
            stop_progress.set()
            progress_thread.join()
            
            # Add a newline after our counter
            sys.stdout.write("\n")
            sys.stdout.flush()
            raise e
            
        chat_time = time.time() - chat_start
        
        # Calculate token usage
        completion_tokens = count_tokens(resp.choices[0].message.content, chat_model)
        total_tokens = prompt_tokens + completion_tokens
        
        logging.info("\n=== Detailed Timing & Usage ===")
        logging.info(f"Search time: {search_time:.2f} seconds")
        logging.info(f"Search tokens: {query_tokens:,} tokens")
        logging.info(f"Chat time: {chat_time:.2f} seconds")
        logging.info(f"Chat tokens (prompt): {prompt_tokens:,} tokens")
        logging.info(f"Chat tokens (completion): {completion_tokens:,} tokens")
        logging.info(f"Chat tokens (total): {total_tokens:,} tokens")
        return {
            'answer': resp.choices[0].message.content,
            'context_files': context_files,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'search_time': search_time,
            'chat_time': chat_time
        }
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

    # Calculate and show input tokens
    prompt_tokens = sum(count_tokens(m["content"], chat_model) for m in messages)
    
    # Check if we're going to exceed the model's token limit
    token_limit = get_model_token_limit(chat_model)
    if prompt_tokens > token_limit:
        logging.error(f"Chat prompt exceeds the token limit for model {chat_model}.")
        logging.error(f"Token count: {prompt_tokens:,}, limit: {token_limit:,}")
        logging.error("\nSuggestions:")
        logging.error("1. Use -n/--no-context flag to chat without code context")
        logging.error("2. Use a smaller directory with 'cq dump <smaller_dir>'")
        logging.error("3. Use a model with larger context (e.g. --model 'openai:gpt-4-turbo')")
        logging.error("4. Use -w/--max-window to limit context tokens (e.g. -w 4000)")
        return None
    
    # Show context info AFTER files list but BEFORE API call
    logging.info(f"Input tokens: {prompt_tokens:,}")
    logging.info("Sending request to OpenAI API...")
    
    # Setup progress indicator thread
    stop_progress = threading.Event()
    request_start = time.time()
    
    # Thread function to show progress while waiting
    def show_progress():
        while not stop_progress.is_set():
            elapsed = time.time() - request_start
            sys.stdout.write(f"\rWaiting for response... [{elapsed:.1f}s]")
            sys.stdout.flush()
            time.sleep(1.0)
    
    # Start progress thread
    progress_thread = threading.Thread(target=show_progress)
    progress_thread.daemon = True
    progress_thread.start()
    
    # Make the API call
    try:
        # Standard call for non-o3-mini models
        client = OpenAI()
        resp = client.chat.completions.create(model=chat_model, messages=messages)
        
        # Stop the progress indicator
        stop_progress.set()
        progress_thread.join()
        
        # Add a newline after our counter
        sys.stdout.write("\n")
        sys.stdout.flush()
        
    except Exception as e:
        # Stop the progress indicator
        stop_progress.set()
        progress_thread.join()
        
        # Add a newline after our counter
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise e
        
    chat_time = time.time() - chat_start
    
    # Calculate token usage
    completion_tokens = count_tokens(resp.choices[0].message.content, chat_model)
    total_tokens = prompt_tokens + completion_tokens
    
    logging.info("\n=== Detailed Timing & Usage ===")
    logging.info(f"Search time: {search_time:.2f} seconds")
    logging.info(f"Search tokens: {query_tokens:,} tokens")
    logging.info(f"Chat time: {chat_time:.2f} seconds")
    logging.info(f"Chat tokens (prompt): {prompt_tokens:,} tokens")
    logging.info(f"Chat tokens (completion): {completion_tokens:,} tokens")
    logging.info(f"Chat tokens (total): {total_tokens:,} tokens")
    
    return {
        'answer': resp.choices[0].message.content,
        'context_files': context_files,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
        'search_time': search_time,
        'chat_time': chat_time
    }
