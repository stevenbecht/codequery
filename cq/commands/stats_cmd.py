import sys
import logging
import openai
import os

from cq.config import load_config
from .util import get_qdrant_client

def register_subparser(subparsers):
    """
    Register the 'stats' subcommand and its arguments.
    """
    stats_parser = subparsers.add_parser("stats", help="Show stats or debug info.")

    stats_parser.add_argument(
        "--all-collections", action="store_true",
        help="If set, show stats for all Qdrant collections instead of just one."
    )
    stats_parser.add_argument(
        "-c", "--collection", type=str, default=None,
        help="Name of the Qdrant collection to show stats for. Defaults to basename(pwd)_collection."
    )
    stats_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    stats_parser.set_defaults(func=handle_stats)

def handle_stats(args):
    """
    Stats subcommand: gather Qdrant stats for either one or all collections.
    """
    config = load_config()

    if not openai.api_key:
        logging.error("No valid OPENAI_API_KEY set. Please update your .env or environment.")
        sys.exit(1)

    client = get_qdrant_client(config["qdrant_host"], config["qdrant_port"], args.verbose)

    if args.all_collections:
        # Show stats for every known collection
        collections_info = client.get_collections()
        all_names = [c.name for c in collections_info.collections]
        logging.info(f"[Stats] Found {len(all_names)} collections: {all_names}")
        for coll_name in all_names:
            _show_stats_for_collection(client, coll_name)
        return
    else:
        # Single collection
        if args.collection:
            coll_name = args.collection
        else:
            pwd_base = os.path.basename(os.getcwd())
            coll_name = pwd_base + "_collection"
        _show_stats_for_collection(client, coll_name)


def _show_stats_for_collection(client, coll_name: str):
    if not client.collection_exists(coll_name):
        logging.info(f"[Stats] Collection '{coll_name}' does not exist.")
        return

    info = client.get_collection(collection_name=coll_name)
    estimated_count = info.points_count or 0

    all_points = []
    limit = 100
    offset = 0
    count_scrolled = 0

    while True:
        points_batch, next_offset = client.scroll(
            collection_name=coll_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        all_points.extend(points_batch)
        count_scrolled += len(points_batch)

        logging.info(f"[Stats] Scrolled {count_scrolled} / {estimated_count} points in '{coll_name}'...")
        if next_offset is None:
            break
        offset = next_offset

    if not all_points:
        logging.info(f"[Stats] No data found in collection '{coll_name}'.")
        return

    file_stats = {}
    total_tokens = 0
    largest_chunks = []

    for record in all_points:
        payload = record.payload or {}
        file_path = payload.get("file_path", "unknown_file")
        chunk_tokens = payload.get("chunk_tokens", 0)
        total_tokens += chunk_tokens

        if file_path not in file_stats:
            file_stats[file_path] = {"chunk_count": 0, "token_sum": 0}
        file_stats[file_path]["chunk_count"] += 1
        file_stats[file_path]["token_sum"] += chunk_tokens

        largest_chunks.append((chunk_tokens, record))

    largest_chunks.sort(key=lambda x: x[0], reverse=True)

    logging.info(f"\n[Stats] Collection '{coll_name}' Summary:")
    logging.info(f" - Total chunks (points): {len(all_points)}")
    logging.info(f" - Total tokens: {total_tokens}")
    avg_tokens = total_tokens / len(all_points) if all_points else 0
    logging.info(f" - Average tokens per chunk: {avg_tokens:.1f}")

    logging.info("\nPer-file Stats:")
    for fpath, stats in file_stats.items():
        logging.info(f"  {fpath}")
        logging.info(f"    Chunks: {stats['chunk_count']}")
        logging.info(f"    Tokens: {stats['token_sum']}\n")

    logging.info("Largest Chunks by Token Count (Top 5):")
    top_5 = largest_chunks[:5]
    for rank, (tok_count, rec) in enumerate(top_5, start=1):
        pl = rec.payload or {}
        fn = pl.get("function_name", "unknown_func")
        start_line = pl.get("start_line", 0)
        end_line = pl.get("end_line", 0)
        logging.info(f" {rank}. {tok_count} tokens | {pl.get('file_path','')} : {fn} (lines {start_line}-{end_line})")

    logging.info("\nDone.\n")
