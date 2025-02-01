import sys
import logging
import openai

from cq.config import load_config
from .util import get_qdrant_client

def register_subparser(subparsers):
    """
    Register the 'stats' subcommand and its arguments.
    """
    stats_parser = subparsers.add_parser("stats", help="Show stats or debug info.")
    stats_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    stats_parser.set_defaults(func=handle_stats)

def handle_stats(args):
    """
    Stats subcommand: gather Qdrant stats with a simple progress display.
    """
    config = load_config()

    if not openai.api_key:
        logging.error("No valid OPENAI_API_KEY set. Please update your .env or environment.")
        sys.exit(1)

    client = get_qdrant_client(config["qdrant_host"], config["qdrant_port"], args.verbose)
    coll = config["qdrant_collection"]

    if not client.collection_exists(coll):
        logging.info(f"[Stats] Collection '{coll}' does not exist.")
        return

    info = client.get_collection(collection_name=coll)
    estimated_count = info.points_count
    if estimated_count is None:
        estimated_count = 0

    all_points = []
    limit = 100
    offset = 0
    count_scrolled = 0

    while True:
        points_batch, next_offset = client.scroll(
            collection_name=coll,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        all_points.extend(points_batch)
        count_scrolled += len(points_batch)

        logging.info(f"[Stats] Scrolled {count_scrolled} / {estimated_count} points...")
        if next_offset is None:
            break
        offset = next_offset

    if not all_points:
        logging.info(f"[Stats] No data found in collection '{coll}'.")
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

    logging.info(f"\n[Stats] Collection '{coll}' Summary:")
    logging.info(f" - Total chunks (points): {len(all_points)}")
    logging.info(f" - Total tokens: {total_tokens}")
    avg_tokens = total_tokens / len(all_points)
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
        logging.info(f" {rank}. {tok_count} tokens | {pl['file_path']} : {fn} (lines {start_line}-{end_line})")

    logging.info("\nDone.\n")
