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
        help="Recursively embed .py and other recognized files in subdirectories."
    )
    embed_parser.add_argument(
        "-c", "--collection", type=str, default=None,
        help="Name of the Qdrant collection. Defaults to basename(pwd)_collection."
    )
    embed_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )

    # NEW: --list and --dump
    embed_parser.add_argument(
        "-l", "--list", action="store_true",
        help="List all files/snippets currently in the collection (skip embedding)."
    )
    embed_parser.add_argument(
        "--dump", action="store_true",
        help="Used with --list to also dump the entire file content from your local codebase."
    )

    embed_parser.set_defaults(func=handle_embed)


def handle_embed(args):
    """
    Embeds directories into Qdrant. By default, it does an incremental embed.

    - If --delete is passed, we delete the collection and stop (no embedding).
      * If not also --force, prompt for confirmation first.

    - If --recreate is passed, we do a fresh index (delete+create+embed).

    - If --list is passed, we skip embedding and simply list everything in the collection
      (and optionally dump file contents if --dump is also set).
    """
    config = load_config()

    if not openai.api_key:
        logging.error("No valid OPENAI_API_KEY set. Please update your .env or environment.")
        sys.exit(1)

    # Determine collection name
    if args.collection:
        collection_name = args.collection
    else:
        pwd_base = os.path.basename(os.getcwd())
        collection_name = pwd_base + "_collection"

    logging.debug(f"[Embed] Using collection name: {collection_name}")

    # Connect to Qdrant
    client = get_qdrant_client(config["qdrant_host"], config["qdrant_port"], args.verbose)

    ################################################################
    # 1) If --delete => delete collection and exit
    ################################################################
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

    ################################################################
    # 2) If --list => list collection contents and exit
    ################################################################
    if args.list:
        if not client.collection_exists(collection_name):
            logging.info(f"[Embed] Collection '{collection_name}' does not exist. Nothing to list.")
            return

        # Scroll all points in the collection
        all_points = []
        limit = 100
        offset = 0
        info = client.get_collection(collection_name=collection_name)
        estimated_count = info.points_count or 0
        count_scrolled = 0

        while True:
            points_batch, next_offset = client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points_batch)
            count_scrolled += len(points_batch)
            logging.debug(f"[Embed][List] Scrolled {count_scrolled}/{estimated_count} points...")
            if next_offset is None:
                break
            offset = next_offset

        if not all_points:
            logging.info(f"[Embed][List] No snippets found in '{collection_name}'.")
            return

        # Group by file_path to produce a summary like "search -l"
        file_data = {}
        for record in all_points:
            payload = record.payload or {}
            file_path = payload.get("file_path", "unknown_file")
            chunk_tokens = payload.get("chunk_tokens", 0)

            if file_path not in file_data:
                file_data[file_path] = {
                    "total_tokens": 0,
                    "chunk_count": 0,
                }
            file_data[file_path]["total_tokens"] += chunk_tokens
            file_data[file_path]["chunk_count"] += 1

        # Sort by total_tokens descending
        sorted_files = sorted(file_data.items(), key=lambda x: x[1]["total_tokens"], reverse=True)

        logging.info(f"\n=== Files in Collection '{collection_name}' ===")
        # Determine the maximum width for tokens for alignment
        max_token_digits = 0
        for _, data in sorted_files:
            token_str = str(data['total_tokens'])
            if len(token_str) > max_token_digits:
                max_token_digits = len(token_str)

        total_files = len(sorted_files)
        grand_total_tokens = sum(d["total_tokens"] for _, d in sorted_files)
        grand_total_chunks = sum(d["chunk_count"] for _, d in sorted_files)

        for file_path, data in sorted_files:
            tokens_fmt = f"{data['total_tokens']:>{max_token_digits}}"
            logging.info(
                f"Tokens: {tokens_fmt} | Chunks: {data['chunk_count']:3d} | File: {file_path}"
            )

        logging.info(f"\nTotal files: {total_files}, total chunks: {grand_total_chunks}, total tokens: {grand_total_tokens}")

        # If --dump => print entire file content from disk (similar to search -l --dump)
        if args.dump:
            for file_path, _info in sorted_files:
                full_path = os.path.join(os.getcwd(), file_path)
                if not os.path.isfile(full_path):
                    logging.warning(f"[Embed][List] Cannot dump file. Not found on disk: {full_path}")
                    continue

                # Attempt to read and dump
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    logging.warning(f"[Embed][List] Error reading {full_path}: {e}")
                    continue

                print(f"\nBEGIN: {file_path}")
                print(content, end="" if content.endswith("\n") else "\n")
                print(f"END: {file_path}")

        return  # Done listing, no embedding

    ################################################################
    # 3) Otherwise, do the normal embedding logic (incremental or recreate)
    ################################################################

    # If --recreate => forcibly delete then embed anew
    recreate_flag = False
    if args.recreate:
        recreate_flag = True
        logging.info(f"[Embed] Recreating collection '{collection_name}' before embedding...")

    # Embed each directory in turn
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
        # After first directory, if we had recreate=True, reset it
        if recreate_flag:
            recreate_flag = False
