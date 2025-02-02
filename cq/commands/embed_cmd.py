import sys
import logging
import openai
import os

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
        help="Delete the specified collection and exit (no embedding)."
    )
    embed_parser.add_argument(
        "-f", "--force", action="store_true",
        help="Skip confirmation prompt when deleting the collection."
    )
    embed_parser.add_argument(
        "--recreate", action="store_true",
        help="Force re-create the collection from scratch (non-incremental)."
    )
    embed_parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Recursively embed .py files in subdirectories"
    )
    embed_parser.add_argument(
        "-c", "--collection", type=str, default=None,
        help="Name of the Qdrant collection. Defaults to basename(pwd)_collection."
    )
    embed_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )

    embed_parser.set_defaults(func=handle_embed)

def handle_embed(args):
    """
    Embeds directories into Qdrant. By default, it does an incremental embed.
    
    - If --delete is passed, we delete the collection and stop (no embedding).
      * If not also --force, prompt for confirmation first.

    - If --recreate is passed, we do a fresh index (delete+create+embed).
      * This is separate from --delete. If you only pass --delete, we skip embedding.
    """
    config = load_config()

    if not openai.api_key:
        logging.error("No valid OPENAI_API_KEY set. Please update your .env or environment.")
        sys.exit(1)

    # Figure out which collection name to use
    if args.collection:
        collection_name = args.collection
    else:
        pwd_base = os.path.basename(os.getcwd())
        collection_name = pwd_base + "_collection"

    logging.debug(f"[Embed] Using collection name: {collection_name}")

    # Connect to Qdrant
    client = get_qdrant_client(config["qdrant_host"], config["qdrant_port"], args.verbose)

    # 1) If --delete is passed, delete the collection and exit
    if args.delete:
        if client.collection_exists(collection_name):
            if not args.force:
                logging.warning(
                    f"[Embed] WARNING: This will DELETE the entire collection '{collection_name}'."
                )
                confirm = input("Proceed? [y/N] ").strip().lower()
                if confirm not in ("y", "yes"):
                    logging.info("[Embed] Aborting delete. Nothing was changed.")
                    return

            logging.debug(f"[Embed] Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)
            logging.info(f"[Embed] Collection '{collection_name}' deleted successfully.")
        else:
            logging.info(f"[Embed] Collection '{collection_name}' does not exist. Nothing to delete.")
        return  # Stop here; no embedding

    # 2) If --recreate is passed, do a fresh embed from scratch
    #    (equivalent to: if exists => delete, then create anew + embed)
    recreate_flag = False
    if args.recreate:
        recreate_flag = True
        logging.info(f"[Embed] Recreating collection '{collection_name}' before embedding...")

    # 3) Otherwise, default is incremental embed
    # We'll embed each directory in turn, using the index_codebase_in_qdrant logic
    for directory in args.directories:
        logging.info(f"[Embed] Embedding directory: {directory}")
        index_codebase_in_qdrant(
            directory=directory,
            collection_name=collection_name,
            qdrant_client=client,
            embed_model=config["openai_embed_model"],
            verbose=args.verbose,
            recursive=args.recursive,
            max_tokens=1500,
            recreate=recreate_flag
        )
        # After the first directory, if we had recreate=True, it’s done its job.
        # Additional directories remain incremental. So set recreate_flag=False
        # to avoid re-deleting the collection for the subsequent directories
        if recreate_flag:
            recreate_flag = False
