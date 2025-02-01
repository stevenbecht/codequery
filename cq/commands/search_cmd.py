import sys
import logging
import openai

from cq.config import load_config
from cq.search import search_codebase_in_qdrant
from .util import get_qdrant_client

def register_subparser(subparsers):
    """
    Register the 'search' subcommand and its arguments.
    """
    search_parser = subparsers.add_parser("search", help="Search code from Qdrant.")
    search_parser.add_argument(
        "-q", "--query", required=True,
        help="Query text"
    )
    search_parser.add_argument(
        "-n", "--num-results", type=int, default=3,
        help="Number of matches to return (ignored if -t/--threshold is set)"
    )
    search_parser.add_argument(
        "-a", "--all", action="store_true",
        help="Return all matches (up to 1000)"
    )
    search_parser.add_argument(
        "-l", "--list", action="store_true",
        help="List only matching file paths"
    )
    search_parser.add_argument(
        "-t", "--threshold", type=float, default=0.25,
        help="Minimum similarity score threshold (0.0 to 1.0)..."
    )
    search_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    # New: -x / --xml-output
    search_parser.add_argument(
        "-x", "--xml-output", action="store_true",
        help="Output search results in XML (for LLM usage)."
    )
    # New: -p / --include-prompt
    search_parser.add_argument(
        "-p", "--include-prompt", action="store_true",
        help="Include a recommended LLM prompt template above the XML."
    )

    search_parser.set_defaults(func=handle_search)

def handle_search(args):
    """
    Search subcommand: just show Qdrant matches, or produce XML with optional LLM prompt.
    """
    config = load_config()

    if not openai.api_key:
        logging.error("No valid OPENAI_API_KEY set. Please update your .env or environment.")
        sys.exit(1)

    client = get_qdrant_client(config["qdrant_host"], config["qdrant_port"], args.verbose)

    # Set appropriate result limits based on mode and threshold
    if args.threshold > 0 and not args.num_results:
        num_results = 1000  # Use a high limit to get all matches above threshold
    elif args.all:
        num_results = 1000
    elif args.list:
        num_results = 25  # Smaller default for listing mode
    else:
        num_results = args.num_results

    search_results = search_codebase_in_qdrant(
        query=args.query,
        collection_name=config["qdrant_collection"],
        qdrant_client=client,
        embed_model=config["openai_embed_model"],
        top_k=num_results,
        verbose=args.verbose
    )
    results = search_results['points']

    # Filter results by threshold score
    filtered_results = [r for r in results if r.score >= args.threshold]

    if not filtered_results:
        logging.info(f"\nNo results found matching the query with threshold {args.threshold}")
        logging.info(f"Try lowering the threshold (current: {args.threshold})")
        return

    # If --list is specified, show only unique file paths with best match scores
    if args.list:
        file_scores = {}
        for match in filtered_results:
            file_path = match.payload["file_path"]
            if file_path not in file_scores or match.score > file_scores[file_path]:
                file_scores[file_path] = match.score

        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        logging.info(f"\n=== Matching Files (threshold: {args.threshold:.2f}) ===")
        for file_path, score in sorted_files:
            logging.info(f"Score: {score:.3f} | File: {file_path}")
        return

    # If -x / --xml-output is set, print results in XML format for LLM usage:
    if getattr(args, "xml_output", False):
        if getattr(args, "include_prompt", False):
            print("""\
You are helping me update my codebase based on the following <search_results> (from our local CLI).

I want you to produce an XML <diff> with a structure like this:

<diff>
  <change>
    <file_path>relative/path/to/file.py</file_path>
    <start_line>10</start_line>
    <end_line>12</end_line>
    <new_code>print("Hello world!")</new_code>
  </change>
  <!-- possibly more <change> elements -->
</diff>

### Requirements:

1. Do not remove existing comments if they are still relevant.
2. When listing changes, provide the entire file content in <new_code> only if you're making a change in that file.
3. If you delete lines, provide a comment explaining why the deletion was needed.

Now here's the raw XML of matched code snippets:
""")

        print("<search_results>")
        for i, match in enumerate(filtered_results, start=1):
            pl = match.payload
            print(f"  <result index='{i}' score='{match.score:.3f}'>")
            print(f"    <file_path>{pl['file_path']}</file_path>")
            print(f"    <function_name>{pl['function_name']}</function_name>")
            print(f"    <start_line>{pl['start_line']}</start_line>")
            print(f"    <end_line>{pl['end_line']}</end_line>")
            code_escaped = pl['code'].replace('<', '&lt;').replace('>', '&gt;')
            print(f"    <code>{code_escaped}</code>")
            print("  </result>")
        print("</search_results>")
        return

    # Otherwise, normal text-based output
    logging.debug("=== Qdrant Search Results (Verbose) ===")
    if args.verbose:
        total_matched_tokens = 0
        for i, match in enumerate(filtered_results, start=1):
            logging.info(f"\n--- Result #{i} ---")
            logging.info(f"ID: {match.id}")
            logging.info(f"Score: {match.score:.3f}")
            if match.vector is not None:
                logging.debug(f"Vector (first 8 dims): {match.vector[:8]} ...")

            payload = match.payload
            if payload:
                file_path = payload.get("file_path", "unknown_file")
                func_name = payload.get("function_name", "unknown_func")
                start_line = payload.get("start_line", 0)
                end_line = payload.get("end_line", 0)
                code_snippet = payload.get("code", "")
                chunk_tokens = payload.get("chunk_tokens", 0)

                total_matched_tokens += chunk_tokens
                logging.info(f"File: {file_path} | Function: {func_name} (lines {start_line}-{end_line})")
                logging.info(f"Chunk tokens: {chunk_tokens}")
                snippet_print = code_snippet if len(code_snippet) < 200 else code_snippet[:200] + "..."
                logging.info(f"Snippet:\n{snippet_print}\n")

        logging.info(f"\nTotal matched tokens (sum of 'chunk_tokens'): {total_matched_tokens}")
    else:
        logging.info(f"=== Matched Snippets (threshold: {args.threshold:.2f}) ===")
        for i, match in enumerate(filtered_results):
            pl = match.payload
            logging.info(f"\n--- Result #{i+1} ---")
            logging.info(f"Score: {match.score:.3f}")
            logging.info(f"File: {pl['file_path']} | Function: {pl['function_name']}")
            logging.info(f"(lines {pl['start_line']}-{pl['end_line']})\n{pl['code']}\n")

    logging.info("\n=== Token Usage ===")
    logging.info(f"Query tokens: {search_results['query_tokens']:,}")
    logging.info(f"Matched snippet tokens: {search_results['snippet_tokens']:,}")
    logging.info(f"Total tokens: {search_results['total_tokens']:,}")
