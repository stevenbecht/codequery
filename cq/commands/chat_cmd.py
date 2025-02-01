import sys
import logging
import openai
import time

from cq.config import load_config
from cq.search import chat_with_context
from .util import get_qdrant_client

def register_subparser(subparsers):
    """
    Register the 'chat' subcommand and its arguments.
    """
    chat_parser = subparsers.add_parser("chat", help="Search + Chat with retrieved code.")
    chat_parser.add_argument(
        "-q", "--query", help="Query text for the chat (required unless --list-models)."
    )
    chat_parser.add_argument(
        "-n", "--num-results", type=int, default=3,
        help="Number of matches to return"
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

def handle_chat(args):
    """
    Chat subcommand: list models if requested, else do a Q&A chat.
    """
    config = load_config()

    if not openai.api_key:
        logging.error("No valid OPENAI_API_KEY set. Please update your .env or environment.")
        sys.exit(1)

    if args.list_models:
        logging.info("[Chat] Listing OpenAI models...\n")
        try:
            model_data = openai.Model.list()
            for m in model_data["data"]:
                logging.info(m["id"])
        except Exception as e:
            logging.error(f"Error listing models: {e}")
        sys.exit(0)

    if not args.query:
        logging.error("You must provide --query/-q unless using --list-models.")
        sys.exit(1)

    # Check if we have input from stdin (e.g., from a pipe)
    stdin_content = ""
    if not sys.stdin.isatty():
        stdin_content = sys.stdin.read().strip()
        if stdin_content:
            logging.debug("[Chat] Received input from stdin")

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

    client = get_qdrant_client(config["qdrant_host"], config["qdrant_port"], args.verbose)

    # Print model info and reasoning effort if o3 model
    model_name = config["openai_chat_model"]
    logging.info(f"\n=== Using Model: {model_name} ===")
    if model_name.startswith('o3-'):
        logging.info(f"Reasoning Effort: {args.reasoning_effort}")
    logging.info("")

    # Combine query with any stdin content
    full_query = args.query
    if stdin_content:
        full_query = f"Here is the content I'm asking about:\n\n{stdin_content}\n\nMy question: {args.query}"

    start_time = time.time()

    answer = chat_with_context(
        query=full_query,
        collection_name=config["qdrant_collection"],
        qdrant_client=client,
        embed_model=config["openai_embed_model"],
        chat_model=config["openai_chat_model"],
        top_k=args.num_results,
        verbose=args.verbose,
        max_context_tokens=args.max_window,
        reasoning_effort=args.reasoning_effort
    )

    end_time = time.time()
    total_time = end_time - start_time

    logging.info(f"\nTotal time: {total_time:.2f} seconds")
    logging.info("\n=== ChatGPT Answer ===")
    logging.info(answer)
