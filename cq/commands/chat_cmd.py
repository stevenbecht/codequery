# ./cq/commands/chat_cmd.py

import sys
import logging
import openai
from openai import OpenAI
import time
import os
import threading

from cq.config import load_config
from cq.search import chat_with_context
from cq.embedding import count_tokens
from .util import get_qdrant_client

def register_subparser(subparsers):
    """
    Register the 'chat' subcommand and its arguments.
    """
    chat_parser = subparsers.add_parser("chat", help="Search + Chat with retrieved code.")
    
    # Renamed from -n/--num-results to -k/--num-results:
    chat_parser.add_argument(
        "-k", "--num-results", type=int, default=3,
        help="Number of matches to return"
    )

    # Add new -n/--no-context:
    chat_parser.add_argument(
        "-n", "--no-context",
        action="store_true",
        help="Ignore any Qdrant collection code context; just do a direct chat with your query."
    )

    chat_parser.add_argument(
        "-q", "--query",
        help="Query text for the chat (required unless --list-models)."
    )
    chat_parser.add_argument(
        "-m", "--model",
        help="Provider:model (e.g. openai:gpt-4)"
    )
    chat_parser.add_argument(
        "--list-models", action="store_true",
        help="List available OpenAI models and exit (no chat performed)."
    )
    chat_parser.add_argument(
        "-c", "--collection", type=str, default=None,
        help="Name of the Qdrant collection to chat over. Defaults to auto-detect from root_dir or basename(pwd)."
    )
    chat_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    chat_parser.add_argument(
        "-w", "--max-window", type=int, default=3000,
        help="Max tokens to use for code context in the chat prompt"
    )
    chat_parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"], default="medium",
        help="Reasoning effort level for o3-mini model (default: medium)"
    )
    chat_parser.set_defaults(func=handle_chat)

def _find_collection_for_current_dir(client, current_dir):
    """
    Same logic as in search_cmd.py - auto-detect which Qdrant collection
    corresponds to 'current_dir' by checking the 'collection_meta' root_dir.
    """
    try:
        collections_info = client.get_collections()
        all_collections = [c.name for c in collections_info.collections]
        best_match = None
        best_len = 0

        for coll_name in all_collections:
            # Scroll or query to find the special metadata record:
            try:
                points_batch, _ = client.scroll(
                    collection_name=coll_name,
                    limit=1_000,
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
        logging.debug(f"[Chat] Error in _find_collection_for_current_dir: {e}")
        return None

def handle_chat(args):
    """
    Chat subcommand:
      - If --list-models, prints available OpenAI models and exits.
      - Otherwise, either:
        * If --no-context, skip code context from Qdrant and just do a direct chat.
        * Else, check if the target collection exists, then calls chat_with_context().
    """
    config = load_config()

    # Ensure we have an OpenAI key
    if not openai.api_key:
        logging.error("No valid OPENAI_API_KEY set. Please update your .env or environment.")
        sys.exit(1)

    # List models if requested
    if args.list_models:
        logging.info("[Chat] Listing OpenAI models...\n")
        try:
            client = OpenAI()
            model_data = client.models.list()
            for m in model_data.data:
                logging.info(m.id)
        except Exception as e:
            logging.error(f"Error listing models: {e}")
        sys.exit(0)

    # Make sure we have a query unless we're just listing models
    if not args.query and not args.no_context:
        logging.error("You must provide --query/-q unless using --list-models or --no-context.")
        sys.exit(1)

    # If user specified a custom provider:model
    if args.model:
        if ":" not in args.model:
            logging.error("Model must be in 'provider:model' format, e.g. openai:gpt-3.5-turbo.")
            sys.exit(1)
        provider, model_name = args.model.split(":", 1)
        if provider.lower() == "openai":
            logging.debug(f"[Chat] Using custom OpenAI model: {model_name}")
            config["openai_chat_model"] = model_name
        else:
            logging.warning(f"Unknown provider '{provider}'. Only 'openai' is supported right now.")
            sys.exit(1)

    model_name = config["openai_chat_model"]

    ################################################################
    # If --no-context, do a direct chat with OpenAI, skipping Qdrant
    ################################################################
    if args.no_context:
        logging.info(f"=== Using Model: {model_name} === (no context mode)")
        
        # If the user piped content via stdin
        stdin_content = ""
        if not sys.stdin.isatty():
            stdin_content = sys.stdin.read().strip()
            if stdin_content:
                logging.debug("[Chat] Received input from stdin (no context mode)")

        # Combine the stdin snippet with the user-provided query
        full_query = args.query or ""
        if stdin_content:
            if full_query:
                full_query = f"Here is the content I'm asking about:\n\n{stdin_content}\n\nMy question: {full_query}"
            else:
                # If somehow --query wasn't specified, but we have stdin, treat that as the entire prompt
                full_query = stdin_content

        # If there's still no content at all, nothing to do
        if not full_query.strip():
            logging.error("No query text provided via --query or stdin in no-context mode.")
            sys.exit(1)

        start_time = time.time()

        # Basic ChatCompletion with user-provided text only
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": full_query}
        ]
        
        # Provide feedback that we're counting tokens
        logging.info("Counting input tokens... ", extra={"end": ""})
        
        # Calculate input token count
        prompt_tokens = sum(count_tokens(m["content"], model_name) for m in messages)
        
        # Show token count immediately
        logging.info(f"Done! ({prompt_tokens:,} tokens)")
        logging.info("Sending request to OpenAI API...")
        
        # Setup for tracking API request time
        request_start = time.time()
        
        # Create a flag to control the progress indicator thread
        stop_progress = threading.Event()
        
        # Thread function to show progress while waiting
        def show_progress():
            while not stop_progress.is_set():
                elapsed = time.time() - request_start
                sys.stdout.write(f"\rWaiting for response... [{elapsed:.1f}s]")
                sys.stdout.flush()
                time.sleep(1.0)
        
        # Start the progress thread
        progress_thread = threading.Thread(target=show_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        # Make the chat request with progress updates
        client = OpenAI()
        try:
            # Make the actual API call
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages
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
            
            # Make sure to add a newline if there was an error
            sys.stdout.write("\n")
            sys.stdout.flush()
            raise e
        
        end_time = time.time()
        
        # Calculate output token count
        completion_tokens = count_tokens(resp.choices[0].message.content, model_name)
        total_tokens = prompt_tokens + completion_tokens

        # Log timing / usage
        total_time = end_time - start_time
        logging.info(f"Total time: {total_time:.2f} seconds")
        logging.info("=== Detailed Timing & Usage ===")
        logging.info(f"Input tokens: {prompt_tokens:,} tokens")
        logging.info(f"Output tokens: {completion_tokens:,} tokens")
        logging.info(f"Total tokens: {total_tokens:,} tokens")
        logging.info("=== ChatGPT Answer ===")
        logging.info(resp.choices[0].message.content)
        return

    ################################################################
    # Otherwise, proceed with normal context-based Qdrant logic
    ################################################################

    client = get_qdrant_client(config["qdrant_host"], config["qdrant_port"], args.verbose)

    # If user provided a collection, we trust that directly
    if args.collection:
        collection_name = args.collection
        logging.debug(f"[Chat] Using user-provided collection: {collection_name}")
    else:
        # Attempt auto-detection
        auto_coll = _find_collection_for_current_dir(client, os.getcwd())
        if auto_coll:
            collection_name = auto_coll
            logging.debug(f"[Chat] Auto-detected collection '{collection_name}' for current directory.")
        else:
            pwd_base = os.path.basename(os.getcwd())
            collection_name = pwd_base + "_collection"
            logging.debug(f"[Chat] No auto-detected match. Default to: {collection_name}")

    # Check if collection exists (avoid 404 from Qdrant)
    if not client.collection_exists(collection_name):
        logging.info(f"[Chat] Collection '{collection_name}' does not exist.")
        logging.info(f"Try running: cq embed -c {collection_name} [--recreate] to create it first.")
        return  # Gracefully exit

    logging.info(f"=== Using Model: {model_name} ===")
    if model_name.startswith('o3-'):
        logging.info(f"Reasoning Effort: {args.reasoning_effort}")

    # If the user piped content via stdin
    stdin_content = ""
    if not sys.stdin.isatty():
        stdin_content = sys.stdin.read().strip()
        if stdin_content:
            logging.debug("[Chat] Received input from stdin")

    # Combine the stdin snippet with the user-provided query
    full_query = args.query or ""
    if stdin_content:
        if full_query:
            full_query = f"Here is the content I'm asking about:\n\n{stdin_content}\n\nMy question: {full_query}"
        else:
            # If somehow --query wasn't specified, but we have stdin, treat that as the entire prompt
            full_query = stdin_content
    
    start_time = time.time()
    
    # Update: check for empty result from chat_with_context
    result = chat_with_context(
        query=full_query,
        collection_name=collection_name,
        qdrant_client=client,
        embed_model=config["openai_embed_model"],
        chat_model=model_name,
        top_k=args.num_results,
        verbose=args.verbose,
        max_context_tokens=args.max_window,
        reasoning_effort=args.reasoning_effort
    )
    
    end_time = time.time()
    
    # Handle the case where chat_with_context returns None due to errors
    if result is None:
        logging.error("Chat failed due to errors. See above messages for details.")
        sys.exit(1)
    
    total_time = end_time - start_time
    
    logging.info(f"Total time: {total_time:.2f} seconds")
    logging.info("=== Detailed Timing & Usage ===")
    logging.info(f"Input tokens: {result['prompt_tokens']:,} tokens")
    logging.info(f"Output tokens: {result['completion_tokens']:,} tokens")
    logging.info(f"Total tokens: {result['total_tokens']:,} tokens")
    logging.info("=== ChatGPT Answer ===")
    logging.info(result['answer'])
