import argparse
import sys
import logging
import time

# Note: remove direct import of requests.exceptions.ConnectionError if you want,
# but it's fine to keep for Qdrant checks:
from requests.exceptions import ConnectionError as RequestsConnectionError

from qdrant_client import QdrantClient

# Our local modules
from cq.config import load_config
from cq.embedding import index_codebase_in_qdrant
from cq.search import search_codebase_in_qdrant, chat_with_context

def setup_logging(verbose: bool):
    """
    Sets up the root logger level to DEBUG if verbose is True,
    otherwise INFO. Outputs to stdout instead of stderr.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, 
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]  # Explicitly use stdout
    )

def get_qdrant_client(host: str, port: int, verbose: bool = False) -> QdrantClient:
    """
    Attempt to create a QdrantClient.
    If there's a connection issue, log a more descriptive error and exit.
    """
    try:
        client = QdrantClient(host=host, port=port)
        return client
    except RequestsConnectionError as e:
        logging.error(f"Could not connect to Qdrant at {host}:{port}. Is Qdrant running?\nDetails: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected Qdrant connection error: {e}")
        sys.exit(1)

def handle_embed(args):
    """
    Embeds directories into Qdrant.
    If --delete is passed, optionally delete (flush) the Qdrant collection,
    then skip re-embedding unless you prefer a fresh embed. If --incremental
    is passed, only new/changed code is embedded.
    """
    config = load_config()

    import openai
    if not openai.api_key:
        logging.error("No valid OPENAI_API_KEY set. Please update your .env or environment.")
        sys.exit(1)

    client = get_qdrant_client(config["qdrant_host"], config["qdrant_port"], args.verbose)

    # If user wants to delete the collection, do so and skip re-embedding entirely.
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

def handle_search(args):
    """Search subcommand: just show Qdrant matches, or produce XML with optional LLM prompt."""
    config = load_config()
    
    import openai
    if not openai.api_key:
        logging.error("No valid OPENAI_API_KEY set. Please update your .env or environment.")
        sys.exit(1)

    client = get_qdrant_client(config["qdrant_host"], config["qdrant_port"], args.verbose)

    # Set appropriate result limits based on mode and threshold
    if args.threshold > 0:
        num_results = 1000  # Use a high limit to get all matches above threshold
    elif args.all:
        num_results = 1000
    elif args.list:
        num_results = 25  # Smaller default for listing mode
    else:
        num_results = args.num_results

    results = search_codebase_in_qdrant(
        query=args.query,
        collection_name=config["qdrant_collection"],
        qdrant_client=client,
        embed_model=config["openai_embed_model"],
        top_k=num_results,
        verbose=args.verbose
    )

    # Filter results by threshold score
    filtered_results = [r for r in results if r.score >= args.threshold]

    if not filtered_results:
        logging.info(f"\nNo results found matching the query with threshold {args.threshold}")
        logging.info(f"Try lowering the threshold (current: {args.threshold})")
        return

    # If --list is specified, show only unique file paths with their best match score
    if args.list:
        file_scores = {}  # Keep track of best score per file
        for match in filtered_results:
            file_path = match.payload["file_path"]
            if file_path not in file_scores or match.score > file_scores[file_path]:
                file_scores[file_path] = match.score
        
        # Sort by score (highest first)
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        logging.info(f"\n=== Matching Files (threshold: {args.threshold:.2f}) ===")
        for file_path, score in sorted_files:
            logging.info(f"Score: {score:.3f} | File: {file_path}")
        return

    # If -x / --xml-output is set, print results in XML format for LLM usage:
    if getattr(args, "xml_output", False):
        # If also --include-prompt, let's print a recommended prompt
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
3. If you make a change, document in the output why it's necessary and how you verified it will work with our indexing. 
4. If you delete lines, provide a comment explaining why the deletion was needed.

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

    # Otherwise, the normal text-based output below
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

def handle_chat(args):
    """Chat subcommand: list models if requested, else do a Q&A chat."""
    config = load_config()
    
    import openai
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

    # If we have stdin content, prepend it to the query for context
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
    
    logging.info("\n=== Query Timing ===")
    logging.info(f"Total time: {total_time:.2f} seconds")
    logging.info("\n=== ChatGPT Answer ===")
    logging.info(answer)

def handle_stats(args):
    """
    Stats subcommand: gather Qdrant stats with a simple progress display.
    Reinstating this to avoid NameError and make 'stats' subcommand work.
    """
    config = load_config()

    import openai
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

def handle_diff(args):
    """
    Handle the 'diff' subcommand:
      --check-diff: Validate a proposed diff (in XML) against codebase.
      --apply-diff: Apply a proposed diff to the codebase if valid.

    The XML structure is assumed to contain <change> elements referencing:
      - file_path
      - start_line
      - end_line
      - new_code

    We added subcommand-level --verbose so you can do `cq diff -v`:
      - if --check-diff + -v => show lines that would be replaced
      - if --apply-diff + -v => show lines as they are replaced
    """
    import os
    import xml.etree.ElementTree as ET

    logging.info(f"[Diff] Reading diff file: {args.diff_file}")
    if not os.path.isfile(args.diff_file):
        logging.error(f"Diff file not found: {args.diff_file}")
        sys.exit(1)

    # Parse the diff.xml file
    try:
        tree = ET.parse(args.diff_file)
        root = tree.getroot()
    except Exception as e:
        logging.error(f"Failed to parse diff XML: {e}")
        sys.exit(1)

    changes_by_file = {}
    for change_node in root.findall(".//change"):
        file_path_node = change_node.find("file_path")
        start_line_node = change_node.find("start_line")
        end_line_node = change_node.find("end_line")
        new_code_node = change_node.find("new_code")

        if file_path_node is None or file_path_node.text is None:
            logging.error("No <file_path> found within <change>. Skipping.")
            continue
        if start_line_node is None or start_line_node.text is None:
            logging.error("No <start_line> found within <change>. Skipping.")
            continue
        if end_line_node is None or end_line_node.text is None:
            logging.error("No <end_line> found within <change>. Skipping.")
            continue
        if new_code_node is None or new_code_node.text is None:
            logging.error("No <new_code> found within <change>. Skipping.")
            continue

        try:
            start_line = int(start_line_node.text)
            end_line = int(end_line_node.text)
        except ValueError:
            logging.error("start_line or end_line is not an integer. Skipping.")
            continue

        file_path = file_path_node.text
        new_code = new_code_node.text

        if file_path not in changes_by_file:
            changes_by_file[file_path] = []
        changes_by_file[file_path].append({
            "start_line": start_line,
            "end_line": end_line,
            "new_code": new_code
        })

    if not changes_by_file:
        logging.info("[Diff] No valid <change> blocks found in diff file.")
        return

    for file_path, changes in changes_by_file.items():
        full_path = os.path.join(args.codebase_dir, file_path)
        if not os.path.isfile(full_path):
            logging.warning(f"[Diff] File does not exist in codebase: {full_path}")
            continue

        logging.info(f"[Diff] Found changes for file: {file_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            original_lines = f.readlines()

        for ch in changes:
            start, end = ch["start_line"], ch["end_line"]
            if start < 0 or end >= len(original_lines):
                logging.error(
                    f"[Diff] Invalid line range {start}-{end} "
                    f"for file {file_path} (file length={len(original_lines)})."
                )
                if args.check_diff:
                    continue
                else:
                    sys.exit(1)

            new_code_lines = ch["new_code"].split("\n")

            if args.check_diff:
                logging.info(f"[Diff] Check only: validated changes for {file_path}")
                # If -v, show the lines that would be replaced
                if args.verbose:
                    logging.info(f"[Diff][Verbose] Proposed replacement in {file_path}, lines {start}-{end}:")
                    for line in new_code_lines:
                        logging.info(f"    > {line}")
            elif args.apply_diff:
                # If -v, show the lines actually replaced
                if args.verbose:
                    logging.info(f"[Diff][Verbose] Replacing lines {start}-{end} in {file_path} with:")
                    for line in new_code_lines:
                        logging.info(f"    > {line}")

                original_lines[start:end+1] = [line + "\n" for line in new_code_lines]

        # After processing all changes for this file,
        # if we're applying, we write back to disk (unless check-diff).
        if args.apply_diff and not args.check_diff:
            with open(full_path, "w", encoding="utf-8") as f:
                f.writelines(original_lines)
            logging.info(f"[Diff] Applied changes to {file_path} successfully.")

def main():
    parser = argparse.ArgumentParser(prog="cq", description="Code-Query CLI with subcommands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # EMBED
    embed_parser = subparsers.add_parser("embed", help="Embed directories into Qdrant.")
    embed_parser.add_argument("-d", "--directories", nargs="+", default=["."], 
                            help="Directories to embed (default = current dir)")
    embed_parser.add_argument("--delete", action="store_true",
                              help="Delete existing Qdrant collection before embedding.")
    embed_parser.add_argument("-r", "--recursive", action="store_true",
                              help="Recursively embed .py files in subdirectories")
    embed_parser.add_argument("-f", "--force", action="store_true",
                              help="Skip confirmation prompt when deleting the DB.")
    embed_parser.add_argument("--incremental", action="store_true",
                              help="Index new/changed files only, skipping re-embedding unchanged snippets.")
    embed_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    embed_parser.set_defaults(func=handle_embed)

    # SEARCH
    search_parser = subparsers.add_parser("search", help="Search code from Qdrant.")
    search_parser.add_argument("-q", "--query", required=True, help="Query text")
    search_parser.add_argument("-n", "--num-results", type=int, default=3, 
                             help="Number of matches to return (ignored if -t/--threshold is set)")
    search_parser.add_argument("-a", "--all", action="store_true", help="Return all matches (up to 1000)")
    search_parser.add_argument("-l", "--list", action="store_true", help="List only matching file paths")
    search_parser.add_argument("-t", "--threshold", type=float, default=0.25,
                             help="Minimum similarity score threshold (0.0 to 1.0). When set, returns all matches above threshold")
    search_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # New: -x / --xml-output
    search_parser.add_argument("-x", "--xml-output", action="store_true",
                               help="Output search results in XML (for LLM usage).")

    # New: -p / --include-prompt
    search_parser.add_argument("-p", "--include-prompt", action="store_true",
                               help="Include a recommended LLM prompt template above the XML.")

    search_parser.set_defaults(func=handle_search)

    # CHAT
    chat_parser = subparsers.add_parser("chat", help="Search + Chat with retrieved code.")
    chat_parser.add_argument("-q", "--query", help="Query text for the chat (required unless --list-models).")
    chat_parser.add_argument("-n", "--num-results", type=int, default=3, help="Number of matches to return")
    chat_parser.add_argument("-m", "--model", help="Provider:model (e.g. openai:gpt-4)")
    chat_parser.add_argument("--list-models", action="store_true",
                             help="List available OpenAI models and exit (no chat performed).")
    chat_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    chat_parser.add_argument("-w", "--max-window", type=int, default=3000,
                             help="Max tokens to use for code context in the chat prompt")
    chat_parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium",
                             help="Reasoning effort level for o3-mini model (default: medium)")
    chat_parser.set_defaults(func=handle_chat)

    # STATS
    stats_parser = subparsers.add_parser("stats", help="Show stats or debug info.")
    stats_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    stats_parser.set_defaults(func=handle_stats)

    # DIFF
    diff_parser = subparsers.add_parser("diff", help="Check or apply code diffs from an XML file.")
    diff_parser.add_argument("--check-diff", action="store_true", help="Check the proposed diff for validity.")
    diff_parser.add_argument("--apply-diff", action="store_true", help="Apply the proposed diff to the codebase.")
    diff_parser.add_argument("--diff-file", required=True, help="Path to the XML diff file.")
    diff_parser.add_argument("--codebase-dir", required=True, help="Path to the codebase on disk to apply changes.")

    # New: subcommand-level verbose for diff
    diff_parser.add_argument("-v", "--verbose", action="store_true",
                             help="Verbose output for diff. Shows lines replaced if --apply-diff or --check-diff.")
    diff_parser.set_defaults(func=handle_diff)

    args = parser.parse_args()
    # The top-level 'setup_logging' depends on args.verbose for subcommands that define it
    setup_logging(args.verbose)
    args.func(args)
