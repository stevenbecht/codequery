import sys
import logging
import openai
import os
import datetime

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

    # --list and --dump
    embed_parser.add_argument(
        "-l", "--list", action="store_true",
        help="List all files/snippets currently in the collection (skip embedding)."
    )
    embed_parser.add_argument(
        "--dump", action="store_true",
        help="Used with --list to also dump the entire file content from your local codebase."
    )

    embed_parser.set_defaults(func=handle_embed)

def _format_timestamp(ts: float) -> str:
    """Convert a float timestamp to human-readable form."""
    if ts <= 0:
        return "N/A"
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def handle_embed(args):
    """
    Embeds directories into Qdrant. By default, it does an incremental embed.

    - If --delete is passed, delete collection and exit.
    - If --recreate is passed, do a fresh index (delete+create+embed).
    - If --list is passed, list everything in the collection (and optionally dump).
    - Otherwise, do normal embedding.
    """
    config = load_config()

    if not openai.api_key:
        logging.error("No valid OPENAI_API_KEY set. Please update your .env or environment.")
        sys.exit(1)

    if args.collection:
        collection_name = args.collection
    else:
        pwd_base = os.path.basename(os.getcwd())
        collection_name = pwd_base + "_collection"

    logging.debug(f"[Embed] Using collection name: {collection_name}")

    client = get_qdrant_client(config["qdrant_host"], config["qdrant_port"], args.verbose)

    # 1) --delete => remove collection and stop
    if args.delete:
        if client.collection_exists(collection_name):
            if not args.force:
                logging.warning(f"[Embed] WARNING: This will DELETE the entire collection '{collection_name}'.")
                confirm = input("Proceed? [y/N] ").strip().lower()
                if confirm not in ("y", "yes"):
                    logging.info("[Embed] Aborting delete. Nothing was changed.")
                    return
            logging.debug(f"[Embed] Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)
            logging.info(f"[Embed] Collection '{collection_name}' deleted successfully.")
        else:
            logging.info(f"[Embed] Collection '{collection_name}' does not exist. Nothing to delete.")
        return

    # 2) --list => scroll and list contents
    if args.list:
        if not client.collection_exists(collection_name):
            logging.info(f"[Embed] Collection '{collection_name}' does not exist. Nothing to list.")
            return

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

        # Summarize per-file
        file_data = {}
        # We'll also track if any snippet is stale for that file.
        for record in all_points:
            payload = record.payload or {}
            file_path = payload.get("file_path", "unknown_file")
            chunk_tokens = payload.get("chunk_tokens", 0)

            if file_path not in file_data:
                file_data[file_path] = {
                    "total_tokens": 0,
                    "chunk_count": 0,
                    "stale": False,  # We'll mark True if any snippet is stale
                }

            file_data[file_path]["total_tokens"] += chunk_tokens
            file_data[file_path]["chunk_count"] += 1

            # Check if snippet is stale
            db_file_mod_time = payload.get("file_mod_time", 0.0)
            try:
                disk_mod = os.path.getmtime(file_path)
                if disk_mod > db_file_mod_time:
                    file_data[file_path]["stale"] = True
            except Exception:
                # If we can't get disk time, ignore
                pass

        # Sort by total_tokens descending
        sorted_files = sorted(file_data.items(), key=lambda x: x[1]["total_tokens"], reverse=True)

        logging.info(f"\n=== Files in Collection '{collection_name}' ===")
        # Determine alignment for tokens
        max_token_digits = 0
        for _, data in sorted_files:
            tstr = str(data['total_tokens'])
            if len(tstr) > max_token_digits:
                max_token_digits = len(tstr)

        total_files = len(sorted_files)
        grand_total_tokens = sum(d["total_tokens"] for _, d in sorted_files)
        grand_total_chunks = sum(d["chunk_count"] for _, d in sorted_files)

        for file_path, data in sorted_files:
            tokens_fmt = f"{data['total_tokens']:>{max_token_digits}}"
            stale_col = "Y" if data["stale"] else "N"
            logging.info(
                f"Tokens: {tokens_fmt} | Chunks: {data['chunk_count']:3d} | Stale: {stale_col} | File: {file_path}"
            )

        logging.info(f"\nTotal files: {total_files}, total chunks: {grand_total_chunks}, total tokens: {grand_total_tokens}")

        # If not verbose, we're done. Only do snippet-level details if --verbose
        if not args.verbose:
            return

        # Verbose => snippet-level detail
        logging.info(f"\n=== Snippet-level details for collection '{collection_name}' ===")
        for record in all_points:
            pl = record.payload or {}
            file_path = pl.get("file_path","unknown_file")
            start_line = pl.get("start_line", -1)
            end_line = pl.get("end_line", -1)

            db_file_mod_time = pl.get("file_mod_time", 0.0)
            db_chunk_embed_time = pl.get("chunk_embed_time", 0.0)

            logging.info(f"* {file_path} lines {start_line}-{end_line}")
            logging.info(f"  DB file_mod_time: {_format_timestamp(db_file_mod_time)}")
            logging.info(f"  DB chunk_embed_time: {_format_timestamp(db_chunk_embed_time)}")

            try:
                disk_mod = os.path.getmtime(file_path)
                if disk_mod > db_file_mod_time:
                    logging.info("  [WARNING] Snippet is older than file on disk.")
            except Exception:
                logging.info("  [WARNING] Could not get disk file time for comparison.")

        # If --dump => also print entire local file content
        if args.dump:
            for file_path, _info in sorted_files:
                full_path = os.path.join(os.getcwd(), file_path)
                if not os.path.isfile(full_path):
                    logging.warning(f"[Embed][List] Cannot dump file. Not found on disk: {full_path}")
                    continue

                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    logging.warning(f"[Embed][List] Error reading {full_path}: {e}")
                    continue

                print(f"\nBEGIN: {file_path}")
                print(content, end="" if content.endswith("\n") else "\n")
                print(f"END: {file_path}")

        return

    # 3) Normal embedding (incremental or recreate)
    recreate_flag = False
    if args.recreate:
        recreate_flag = True
        logging.info(f"[Embed] Recreating collection '{collection_name}' before embedding...")

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
        # Only recreate once, for the first directory
        if recreate_flag:
            recreate_flag = False
