import sys
import logging
import openai

from cq.config import load_config
from cq.embedding import index_codebase_in_qdrant
from .util import get_qdrant_client

def register_subparser(subparsers):
    """
    Register the 'embed' subcommand and its arguments.
    """
    embed_parser = subparsers.add_parser("embed", help="Embed directories into Qdrant.")
    embed_parser.add_argument(
        "-d", "--directories", nargs="+", default=["."],
        help="Directories to embed (default = current dir)"
    )
    embed_parser.add_argument(
        "--delete", action="store_true",
        help="Delete existing Qdrant collection before embedding."
    )
    embed_parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Recursively embed .py files in subdirectories"
    )
    embed_parser.add_argument(
        "-f", "--force", action="store_true",
        help="Skip confirmation prompt when deleting the DB."
    )
    embed_parser.add_argument(
        "--incremental", action="store_true",
        help="Index new/changed files only, skipping re-embedding unchanged snippets."
    )
    embed_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    embed_parser.set_defaults(func=handle_embed)

def handle_embed(args):
    """
    Embeds directories into Qdrant.
    If --delete is passed, optionally delete (flush) the Qdrant collection,
    then skip re-embedding unless you prefer a fresh embed. If --incremental
    is passed, only new/changed code is embedded.
    """
    config = load_config()

    if not openai.api_key:
        logging.error("No valid OPENAI_API_KEY set. Please update your .env or environment.")
        sys.exit(1)

    client = get_qdrant_client(config["qdrant_host"], config["qdrant_port"], args.verbose)

    # If user wants to delete the collection, do so and skip re-embedding
    if args.delete:
        if not args.force:
            logging.warning(
                f"WARNING: This will DELETE the entire collection '{config['qdrant_collection']}'."
            )
            confirm = input("Proceed? [y/N] ").strip().lower()
            if confirm not in ("y", "yes"):
                logging.info("Aborting delete. Nothing was changed.")
                return

        if client.collection_exists(config["qdrant_collection"]):
            logging.debug(f"Deleting existing collection '{config['qdrant_collection']}'...")
            client.delete_collection(config["qdrant_collection"])
            logging.info(f"Collection '{config['qdrant_collection']}' deleted successfully.")

        logging.info("Delete operation completed. Skipping re-embedding.")
        return

    # If we get here, user is NOT doing a delete-only workflow
    for directory in args.directories:
        logging.info(f"Embedding directory: {directory}")
        index_codebase_in_qdrant(
            directory=directory,
            collection_name=config["qdrant_collection"],
            qdrant_client=client,
            embed_model=config["openai_embed_model"],
            verbose=args.verbose,
            recursive=args.recursive,
            incremental=args.incremental
        )
