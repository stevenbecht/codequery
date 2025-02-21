# BEGIN: ./cq/commands/search_cmd.py
import sys
import logging
import openai
import os
from requests.exceptions import ConnectionError as RequestsConnectionError

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
        help="List only matching file paths (plus token counts)."
    )
    search_parser.add_argument(
        "-d", "--dump", action="store_true",
        help="When used with -l, also print/dump the entire contents of each matching file."
    )
    search_parser.add_argument(
        "-t", "--threshold", type=float, default=0.25,
        help="Minimum similarity score threshold (0.0 to 1.0)."
    )
    search_parser.add_argument(
        "--all-collections", action="store_true",
        help="If set, search across all Qdrant collections, merging results by score."
    )
    search_parser.add_argument(
        "-c", "--collection", type=str, default=None,
        help="Name of a specific Qdrant collection to search. Defaults to basename(pwd)_collection if not set."
    )
    search_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    # -x / --xml-output
    search_parser.add_argument(
        "-x", "--xml-output", action="store_true",
        help="Output search results in XML (for LLM usage)."
    )
    # -p / --include-prompt
    search_parser.add_argument(
        "-p", "--include-prompt", action="store_true",
        help="Include a recommended LLM prompt template above the XML."
    )

    search_parser.set_defaults(func=handle_search)

def guess_language_from_extension(file_path: str) -> str:
    """
    Very simple mapping from extension => "language" label.
    Fallback to "plaintext" if unknown.
    """
    _, ext = os.path.splitext(file_path.lower())
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".cs": "csharp",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".pl": "perl",
        ".rb": "ruby",
        ".php": "php",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "scss",
        ".md": "markdown",
        ".json": "json",
        ".toml": "toml",
        ".ini": "ini",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    return ext_map.get(ext, "plaintext")

def handle_search(args):
    """
    Search subcommand: checks if the collection exists (or multiple),
    then does a Qdrant query.
    """
    config = load_config()

    if not openai.api_key:
        logging.error("No valid OPENAI_API_KEY set. Please update your .env or environment.")
        sys.exit(1)

    if args.threshold < 0 or args.threshold > 1.0:
        logging.error("Threshold must be between 0.0 and 1.0.")
        sys.exit(1)

    client = get_qdrant_client(config["qdrant_host"], config["qdrant_port"], args.verbose)

    # Figure out how many results we want
    if args.threshold > 0 and not args.num_results:
        num_results = 1000
    elif args.all:
        num_results = 1000
    elif args.list:
        num_results = 25
    else:
        num_results = args.num_results

    try:
        ################################################################
        # Collect results from single or multiple collections
        ################################################################
        if args.all_collections:
            collections_info = client.get_collections()
            all_collections = [c.name for c in collections_info.collections]
            if not all_collections:
                logging.info("[Search] No collections exist in Qdrant. Try embedding first.")
                return

            logging.debug(f"[Search] Searching across collections: {all_collections}")

            merged_points = []
            total_query_tokens = 0
            total_snippet_tokens = 0

            # We'll query each collection, then merge results by score
            for coll_name in all_collections:
                sub_results = _safe_search(
                    client=client,
                    collection_name=coll_name,
                    query=args.query,
                    top_k=num_results,
                    embed_model=config["openai_embed_model"],
                    verbose=args.verbose
                )
                if not sub_results:
                    continue  # skip if empty or nonexistent

                # Label each result with the coll_name
                for p in sub_results["points"]:
                    p.payload["collection_name"] = coll_name

                merged_points.extend(sub_results["points"])
                total_query_tokens += sub_results["query_tokens"]
                total_snippet_tokens += sub_results["snippet_tokens"]

            # Sort merged by score desc, keep top N
            merged_points.sort(key=lambda x: x.score, reverse=True)
            if len(merged_points) > num_results:
                merged_points = merged_points[:num_results]

            search_results = {
                "points": merged_points,
                "query_tokens": total_query_tokens,
                "snippet_tokens": total_snippet_tokens,
                "total_tokens": total_query_tokens + total_snippet_tokens,
            }

        else:
            # Single-collection approach
            if args.collection:
                collection_name = args.collection
            else:
                pwd_base = os.path.basename(os.getcwd())
                collection_name = pwd_base + "_collection"

            # Make sure collection exists
            if not client.collection_exists(collection_name):
                logging.info(f"[Search] Collection '{collection_name}' does not exist.")
                logging.info(f"Try running: cq embed -c {collection_name} [--recreate] to create it first.")
                return

            search_results = _safe_search(
                client=client,
                collection_name=collection_name,
                query=args.query,
                top_k=num_results,
                embed_model=config["openai_embed_model"],
                verbose=args.verbose
            )
            if not search_results:
                logging.info("[Search] No data found or collection is empty.")
                return

            # Label the collection if missing
            for p in search_results["points"]:
                if "collection_name" not in p.payload:
                    p.payload["collection_name"] = collection_name

    except openai.error.AuthenticationError:
        logging.error("Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable.")
        sys.exit(1)
    except (openai.error.APIConnectionError, RequestsConnectionError) as e:
        logging.error(f"Connection error: {e}")
        logging.error("Please check your internet connection and ensure required services are running.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)

    points = search_results['points']

    ################################################################
    # Filter results by threshold
    ################################################################
    filtered_results = [r for r in points if r.score >= args.threshold]
    if not filtered_results:
        logging.info(f"\nNo results found matching the query with threshold {args.threshold}")
        logging.info(f"Try lowering the threshold (current: {args.threshold})")
        return

    ################################################################
    # 1) --list mode
    #    a) if --xml-output and --dump => produce <search_results><codefile>...</codefile></search_results>
    #    b) else normal text listing
    ################################################################
    if args.list:
        file_data = {}
        for match in filtered_results:
            file_path = match.payload.get("file_path", "unknown_file")
            chunk_tokens = match.payload.get("chunk_tokens", 0)
            score = match.score

            if file_path not in file_data:
                file_data[file_path] = {
                    "score": score,
                    "tokens": chunk_tokens
                }
            else:
                # Update best score if this snippet's is higher
                if score > file_data[file_path]["score"]:
                    file_data[file_path]["score"] = score
                # Add snippet tokens to the total tokens for that file
                file_data[file_path]["tokens"] += chunk_tokens

        # Sort by best score descending
        sorted_files = sorted(file_data.items(), key=lambda x: x[1]["score"], reverse=True)

        # If user wants XML output for file dump
        if args.xml_output and args.dump:
            # Possibly include a prompt if requested
            if args.include_prompt:
                print("""\
You are helping me by providing an XML dump of the matched files from our local CLI search.

The structure is:

<search_results>
  <codefile>
    <metadata>
      <filename>...</filename>
      <language>...</language>
      <tokens>123</tokens>
      <score>0.450</score>
    </metadata>
    <content>
      <![CDATA[
      ...entire file contents...
      ]]>
    </content>
  </codefile>
  ...
</search_results>
""")

            print("<search_results>")
            for file_path, info in sorted_files:
                full_path = os.path.join(os.getcwd(), file_path)
                language = guess_language_from_extension(file_path)
                # Attempt to read entire file
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                except Exception as e:
                    file_content = f"[ERROR: could not read file: {str(e)}]"

                # Escape any ']]>' inside the content to avoid breaking the CDATA
                # (In practice, you rarely see ']]>' in code, but let's be safe.)
                safe_content = file_content.replace("]]>", "]]]]><![CDATA[>")

                # Build <codefile> element
                print("  <codefile>")
                print("    <metadata>")
                print(f"      <filename>{file_path}</filename>")
                print(f"      <language>{language}</language>")
                print(f"      <tokens>{info['tokens']}</tokens>")
                print(f"      <score>{info['score']:.3f}</score>")
                print("    </metadata>")
                print("    <content><![CDATA[")
                print(safe_content, end="" if safe_content.endswith("\n") else "\n")
                print("]]></content>")
                print("  </codefile>")

            print("</search_results>")

            # That's all – we skip normal listing
            return

        ################################################################
        # Otherwise, if user wants plain text listing or partial dump
        ################################################################

        # Print text-based summary
        logging.info(f"\n=== Matching Files (threshold: {args.threshold:.2f}) ===")

        # Determine max width of the 'tokens' column for alignment
        max_token_digits = 0
        for _, info in sorted_files:
            token_str = str(info['tokens'])
            if len(token_str) > max_token_digits:
                max_token_digits = len(token_str)

        total_tokens_across_files = 0
        for file_path, info in sorted_files:
            total_tokens_across_files += info["tokens"]
            # Format tokens with right alignment
            tokens_formatted = f"{info['tokens']:>{max_token_digits}}"
            logging.info(
                f"Score: {info['score']:.3f} | Tokens: {tokens_formatted} | File: {file_path}"
            )

        logging.info(f"\n=== Total tokens (across all matched files): {total_tokens_across_files} ===")

        # If user also specified --dump but not --xml-output, do the "BEGIN/END" style:
        if args.dump and not args.xml_output:
            for file_path, info in sorted_files:
                full_path = os.path.join(os.getcwd(), file_path)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                except Exception as e:
                    file_content = f"[ERROR: could not read file: {str(e)}]"

                print(f"\nBEGIN: {file_path}")
                print(file_content, end="" if file_content.endswith("\n") else "\n")
                print(f"END: {file_path}")

        return

    ################################################################
    # 2) If -x / --xml-output but not in -l mode => standard snippet-based XML
    ################################################################
    if args.xml_output:
        if args.include_prompt:
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
            print(f"    <collection_name>{pl.get('collection_name','')}</collection_name>")
            print(f"    <file_path>{pl.get('file_path','')}</file_path>")
            print(f"    <function_name>{pl.get('function_name','')}</function_name>")
            print(f"    <start_line>{pl.get('start_line','')}</start_line>")
            print(f"    <end_line>{pl.get('end_line','')}</end_line>")
            code_escaped = pl.get('code','').replace('<', '&lt;').replace('>', '&gt;')
            print(f"    <code>{code_escaped}</code>")
            print("  </result>")
        print("</search_results>")
        return

    ################################################################
    # 3) Normal text-based snippet details
    ################################################################
    logging.debug("=== Qdrant Search Results (Verbose) ===")
    if args.verbose:
        total_matched_tokens = 0
        for i, match in enumerate(filtered_results, start=1):
            logging.info(f"\n--- Result #{i} ---")

            pl = match.payload
            coll_label = pl.get("collection_name", "unknown_collection")
            logging.info(f"Collection: {coll_label}")
            logging.info(f"Score: {match.score:.3f}")

            if match.vector is not None:
                logging.debug(f"Vector (first 8 dims): {match.vector[:8]} ...")

            if pl:
                file_path = pl.get("file_path", "unknown_file")
                func_name = pl.get("function_name", "unknown_func")
                start_line = pl.get("start_line", 0)
                end_line = pl.get("end_line", 0)
                code_snippet = pl.get("code", "")
                chunk_tokens = pl.get("chunk_tokens", 0)

                total_matched_tokens += chunk_tokens
                logging.info(f"File: {file_path} | Function: {func_name} (lines {start_line}-{end_line})")
                logging.info(f"Chunk tokens: {chunk_tokens}")
                snippet_print = code_snippet if len(code_snippet) < 200 else code_snippet[:200] + "..."
                logging.info(f"Snippet:\n{snippet_print}\n")

        logging.info(f"\nTotal matched tokens (sum of 'chunk_tokens'): {total_matched_tokens}")
    else:
        logging.info(f"=== Matched Snippets (threshold: {args.threshold:.2f}) ===")
        for i, match in enumerate(filtered_results, start=1):
            pl = match.payload
            logging.info(f"\n--- Result #{i} ---")

            coll_label = pl.get("collection_name", "unknown_collection")
            logging.info(f"Collection: {coll_label}")
            logging.info(f"Score: {match.score:.3f}")

            file_path = pl.get("file_path", "unknown_file")
            func_name = pl.get("function_name", "unknown_func")
            start_line = pl.get("start_line", 0)
            end_line = pl.get("end_line", 0)
            code_snippet = pl.get("code", "")

            logging.info(f"File: {file_path} | Function: {func_name}")
            logging.info(f"(lines {start_line}-{end_line})\n{code_snippet}\n")

    logging.info("\n=== Token Usage ===")
    logging.info(f"Query tokens: {search_results['query_tokens']:,}")
    logging.info(f"Matched snippet tokens: {search_results['snippet_tokens']:,}")
    logging.info(f"Total tokens: {search_results['total_tokens']:,}")

def _safe_search(client, collection_name, query, top_k, embed_model, verbose=False):
    """
    Helper function to do the actual Qdrant query,
    returning None if the collection is empty or doesn't exist.
    """
    if not client.collection_exists(collection_name):
        logging.info(f"[Search] Collection '{collection_name}' does not exist. Skipping.")
        return None

    try:
        return search_codebase_in_qdrant(
            query=query,
            collection_name=collection_name,
            qdrant_client=client,
            embed_model=embed_model,
            top_k=top_k,
            verbose=verbose
        )
    except Exception as e:
        logging.warning(f"[Search] Error searching collection '{collection_name}': {e}")
        return None
# END: ./cq/commands/search_cmd.py
