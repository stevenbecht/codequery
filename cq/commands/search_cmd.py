import sys
import logging
import openai
from openai import OpenAI
import os
import datetime
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
        help="Name of a specific Qdrant collection to search. Defaults to auto-detect from root_dir or basename(pwd)."
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

def _find_collection_for_current_dir(client, current_dir):
    """
    Attempt to auto-detect which Qdrant collection corresponds
    to 'current_dir' by checking any 'collection_meta' root_dir
    payload in each collection. We pick the one for which
    root_dir is an ancestor of current_dir (commonpath == root_dir).
    If multiple match, pick the longest root_dir.
    Return the matching collection name, or None if none found.
    """
    try:
        collections_info = client.get_collections()
        all_collections = [c.name for c in collections_info.collections]
        best_match = None
        best_len = 0

        for coll_name in all_collections:
            # Scroll or query to find the special metadata record:
            try:
                points_batch, _ = client.scroll(
                    collection_name=coll_name,
                    limit=1_000,  # in small codebases, 1000 is enough to find id=0 easily
                    with_payload=True,
                    with_vectors=False
                )
                for p in points_batch:
                    pl = p.payload
                    if not pl:
                        continue
                    # Check if this is the "collection_meta" record
                    if pl.get("collection_meta") and "root_dir" in pl:
                        root_dir = os.path.abspath(pl["root_dir"])
                        cur_dir_abs = os.path.abspath(current_dir)
                        common = os.path.commonpath([root_dir, cur_dir_abs])
                        if common == root_dir:
                            # current_dir is inside root_dir or equal to it
                            rlen = len(root_dir)
                            if rlen > best_len:
                                best_len = rlen
                                best_match = coll_name
            except:
                pass

        return best_match
    except Exception as e:
        logging.debug(f"[Search] Error in _find_collection_for_current_dir: {e}")
        return None

def handle_search(args):
    """
    Search subcommand: checks if the collection exists (or multiple),
    then does a Qdrant query, prints results (including STALE columns).
    """
    config = load_config()

    if not openai.api_key:
        logging.error("No valid OPENAI_API_KEY set. Please update your .env or environment.")
        sys.exit(1)

    if args.threshold < 0 or args.threshold > 1.0:
        logging.error("Threshold must be between 0.0 and 1.0.")
        sys.exit(1)

    client = get_qdrant_client(config["qdrant_host"], config["qdrant_port"], args.verbose)

    # Determine how many results to request
    if args.threshold > 0 and not args.num_results:
        num_results = 1000
    elif args.all:
        num_results = 1000
    elif args.list and not args.num_results:  # Only use 25 if user didn't specify -n
        num_results = 25
    else:
        num_results = args.num_results

    # If user wants to search all collections, handle that:
    if args.all_collections:
        try:
            collections_info = client.get_collections()
            all_collections = [c.name for c in collections_info.collections]
            if not all_collections:
                logging.info("[Search] No collections exist in Qdrant. Try embedding first.")
                return

            logging.debug(f"[Search] Searching across collections: {all_collections}")

            merged_points = []
            total_query_tokens = 0
            total_snippet_tokens = 0

            # Query each collection, then merge
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
                    continue

                # Label each result with the coll_name
                for p in sub_results["points"]:
                    p.payload["collection_name"] = coll_name

                merged_points.extend(sub_results["points"])
                total_query_tokens += sub_results["query_tokens"]
                total_snippet_tokens += sub_results["snippet_tokens"]

            # Sort by score desc and keep top
            merged_points.sort(key=lambda x: x.score, reverse=True)
            if len(merged_points) > num_results:
                merged_points = merged_points[:num_results]

            search_results = {
                "points": merged_points,
                "query_tokens": total_query_tokens,
                "snippet_tokens": total_snippet_tokens,
                "total_tokens": total_query_tokens + total_snippet_tokens,
            }

        except openai.AuthenticationError:
            logging.error("Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable.")
            sys.exit(1)
        except (openai.APIConnectionError, RequestsConnectionError) as e:
            logging.error(f"Connection error: {e}")
            logging.error("Please check your internet connection and ensure required services are running.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            sys.exit(1)

    else:
        # Single collection => auto-detect or user-specified
        if args.collection:
            collection_name = args.collection
            logging.debug(f"[Search] Using user-provided collection name: {collection_name}")
        else:
            auto_coll = _find_collection_for_current_dir(client, os.getcwd())
            if auto_coll:
                collection_name = auto_coll
                logging.debug(f"[Search] Auto-detected collection '{collection_name}' for current directory.")
            else:
                pwd_base = os.path.basename(os.getcwd())
                collection_name = pwd_base + "_collection"
                logging.debug(f"[Search] No auto-detected match. Default to: {collection_name}")

        try:
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

        except openai.AuthenticationError:
            logging.error("Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable.")
            sys.exit(1)
        except (openai.APIConnectionError, RequestsConnectionError) as e:
            logging.error(f"Connection error: {e}")
            logging.error("Please check your internet connection and ensure required services are running.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            sys.exit(1)

    points = search_results['points']

    # Filter results by threshold
    filtered_results = [r for r in points if r.score >= args.threshold]
    if not filtered_results:
        logging.info(f"\nNo results found matching the query with threshold {args.threshold}")
        logging.info(f"Try lowering the threshold (current: {args.threshold})")
        return

    # --------------------------------------------------------------------------
    # LIST MODE
    # --------------------------------------------------------------------------
    if args.list:
        # We'll collect data per file, track best score, sum tokens, check stale
        file_data = {}
        for match in filtered_results:
            file_path = match.payload.get("file_path", "unknown_file")
            chunk_tokens = match.payload.get("chunk_tokens", 0)
            score = match.score
            db_file_mod = match.payload.get("file_mod_time", 0.0)

            # Initialize if not present
            if file_path not in file_data:
                file_data[file_path] = {
                    "score": score,
                    "tokens": chunk_tokens,
                    "stale": False,
                    "chunks": 1  # Track number of chunks per file
                }
            else:
                # Update best score if higher
                if score > file_data[file_path]["score"]:
                    file_data[file_path]["score"] = score
                file_data[file_path]["tokens"] += chunk_tokens
                file_data[file_path]["chunks"] += 1  # Increment chunk count

            # Check stale
            try:
                disk_mod = os.path.getmtime(file_path)
                # Use a small tolerance (0.001 seconds) to avoid floating point precision issues
                if disk_mod - db_file_mod > 0.001:  
                    file_data[file_path]["stale"] = True
            except Exception:
                # Ignore if we can't stat the file
                pass

        # Sort by best score
        sorted_files = sorted(file_data.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Get total count before limiting
        total_matching_files = len(sorted_files)
        
        # Limit number of files based on -n parameter, unless --all is specified
        if not args.all and args.num_results > 0:
            original_count = len(sorted_files)
            sorted_files = sorted_files[:args.num_results]
            if len(sorted_files) < original_count:
                logging.debug(f"[Search][List] Limited output from {original_count} to {args.num_results} files due to -n parameter")

        # If user wants XML + dump entire files
        if args.xml_output and args.dump:
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
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                except Exception as e:
                    file_content = f"[ERROR: could not read file: {str(e)}]"

                safe_content = file_content.replace("]]>", "]]]]><![CDATA[>")

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
            return

        # Otherwise, plain text listing
        file_count = len(sorted_files)
        if file_count < total_matching_files:
            logging.info(f"=== Matching Files (showing {file_count} of {total_matching_files}) (threshold: {args.threshold:.2f}) ===")
        else:
            logging.info(f"=== Matching Files ({file_count}) (threshold: {args.threshold:.2f}) ===")

        # Figure out alignment for tokens
        max_token_digits = 0
        for _, info in sorted_files:
            tstr = str(info['tokens'])
            if len(tstr) > max_token_digits:
                max_token_digits = len(tstr)

        total_tokens_across_files = 0
        for file_path, info in sorted_files:
            total_tokens_across_files += info["tokens"]
            tokens_formatted = f"{info['tokens']:>{max_token_digits}}"
            stale_col = "Y" if info["stale"] else "N"
            chunk_info = f" ({info['chunks']} chunks)" if info['chunks'] > 1 else ""
            logging.info(
                f"Score: {info['score']:.3f} | Tokens: {tokens_formatted} | Stale: {stale_col} | File: {file_path}{chunk_info}"
            )

        logging.info(f"=== Total tokens (across {file_count} matched files): {total_tokens_across_files} ===")
        
        # If showing limited results, remind user of total matches
        if file_count < total_matching_files:
            logging.info(f"=== Showing {file_count} of {total_matching_files} total matching files (use --all to show all) ===")
        
        # If any file is stale, show the warning message
        if any(info["stale"] for _, info in sorted_files):
            logging.warning("=" * 80)
            logging.warning("STALE ENTRIES FOUND - UPDATE EMBEDDINGS FOR BEST RESULTS")
            logging.warning("RUN: cq embed --recreate -r -d .")
            logging.warning("=" * 80)

        # If not verbose => done.  If --dump && not XML => do "BEGIN/END" file contents
        if not args.verbose and not args.dump:
            return

        # Verbose => show snippet-level detail
        if args.verbose:
            logging.info("=== Snippet-level details ===")
            # We'll re-loop over filtered_results and print timestamps
            for match in filtered_results:
                pl = match.payload
                file_path = pl.get("file_path", "unknown_file")
                start_line = pl.get("start_line", 0)
                end_line = pl.get("end_line", 0)
                code_snippet = pl.get("code", "")
                db_file_mod = pl.get("file_mod_time", 0)
                db_chunk_embed = pl.get("chunk_embed_time", 0)

                logging.info(f"File: {file_path}, lines {start_line}-{end_line}")
                logging.info(f"Score: {match.score:.3f}")
                snippet_print = code_snippet if len(code_snippet) < 200 else code_snippet[:200] + "..."
                logging.info(f"Snippet:\n{snippet_print}")

                file_mod_dt = datetime.datetime.fromtimestamp(db_file_mod).isoformat() if db_file_mod else "N/A"
                chunk_embed_dt = datetime.datetime.fromtimestamp(db_chunk_embed).isoformat() if db_chunk_embed else "N/A"
                logging.info(f"DB file_mod_time: {file_mod_dt}, chunk_embed_time: {chunk_embed_dt}")

                try:
                    disk_mod = os.path.getmtime(file_path)
                    if disk_mod > db_file_mod:
                        logging.info("[WARNING] This snippet may be outdated (disk mod time is newer).")
                except Exception as e:
                    logging.debug(f"Could not get disk mod time for {file_path}: {e}")

                logging.info("-------------------------")

        # If user also specified --dump => do a "BEGIN/END" for each file
        if args.dump:
            file_paths_in_results = list(set(match.payload.get("file_path", "unknown_file") for match in filtered_results))
            for file_path in file_paths_in_results:
                full_path = os.path.join(os.getcwd(), file_path)
                if not os.path.isfile(full_path):
                    logging.warning(f"[Search][List] Cannot dump file. Not on disk: {full_path}")
                    continue
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    logging.warning(f"[Search][List] Error reading {full_path}: {e}")
                    continue

                print(f"\nBEGIN: {file_path}")
                print(content, end="" if content.endswith("\n") else "\n")
                print(f"END: {file_path}")

        return

    # --------------------------------------------------------------------------
    # NON-LIST MODE => snippet results
    # --------------------------------------------------------------------------
    if args.xml_output:
        # XML snippet-level
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
            # Timestamps
            file_mod_time = pl.get("file_mod_time", 0)
            chunk_embed_time = pl.get("chunk_embed_time", 0)
            print(f"    <file_mod_time>{file_mod_time}</file_mod_time>")
            print(f"    <chunk_embed_time>{chunk_embed_time}</chunk_embed_time>")
            print("  </result>")
        print("</search_results>")
        return

    # Normal text-based snippet results
    logging.debug("=== Qdrant Search Results (Verbose) ===")
    if args.verbose:
        total_matched_tokens = 0
        for i, match in enumerate(filtered_results, start=1):
            logging.info(f"--- Result #{i} ---")
            logging.info(f"Score: {match.score:.3f}")

            pl = match.payload
            coll_label = pl.get("collection_name", "unknown_collection")
            logging.info(f"Collection: {coll_label}")

            if match.vector is not None:
                logging.debug(f"Vector (first 8 dims): {match.vector[:8]} ...")

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
            logging.info(f"Snippet:\n{snippet_print}")

            db_file_mod = pl.get("file_mod_time", 0)
            db_chunk_embed = pl.get("chunk_embed_time", 0)
            file_mod_dt = datetime.datetime.fromtimestamp(db_file_mod).isoformat() if db_file_mod else "N/A"
            chunk_embed_dt = datetime.datetime.fromtimestamp(db_chunk_embed).isoformat() if db_chunk_embed else "N/A"
            logging.info(f"Database file_mod_time: {file_mod_dt}")
            logging.info(f"Chunk embed time: {chunk_embed_dt}")

            try:
                disk_mod = os.path.getmtime(file_path)
                if disk_mod > db_file_mod:
                    logging.info("[WARNING] This snippet may be outdated (disk mod time is newer).")
            except Exception as e:
                logging.debug(f"Could not get disk mod time for {file_path}: {e}")

            logging.info("-------------------------")

        logging.info(f"Total matched tokens (sum of 'chunk_tokens'): {total_matched_tokens}")
    else:
        logging.info("=== Matched Snippets (threshold: {args.threshold:.2f}) ===")
        for i, match in enumerate(filtered_results, start=1):
            pl = match.payload
            logging.info(f"--- Result #{i} ---")
            logging.info(f"Score: {match.score:.3f}")

            coll_label = pl.get("collection_name", "unknown_collection")
            logging.info(f"Collection: {coll_label}")

            file_path = pl.get("file_path", "unknown_file")
            func_name = pl.get("function_name", "unknown_func")
            start_line = pl.get("start_line", 0)
            end_line = pl.get("end_line", 0)
            code_snippet = pl.get("code", "")

            logging.info(f"File: {file_path} | Function: {func_name}")
            logging.info(f"(lines {start_line}-{end_line})\n{code_snippet}")

            db_file_mod = pl.get("file_mod_time", 0)
            db_chunk_embed = pl.get("chunk_embed_time", 0)
            file_mod_dt = datetime.datetime.fromtimestamp(db_file_mod).isoformat() if db_file_mod else "N/A"
            chunk_embed_dt = datetime.datetime.fromtimestamp(db_chunk_embed).isoformat() if db_chunk_embed else "N/A"
            logging.info(f"Database file_mod_time: {file_mod_dt}")
            logging.info(f"Chunk embed time: {chunk_embed_dt}")

            try:
                disk_mod = os.path.getmtime(file_path)
                if disk_mod > db_file_mod:
                    logging.info("[WARNING] This snippet may be outdated (disk mod time is newer).")
            except Exception as e:
                logging.debug(f"Could not get disk mod time for {file_path}: {e}")

    logging.info("=== Token Usage ===")
    logging.info(f"Query tokens: {search_results['query_tokens']:,}")
    logging.info(f"Matched snippet tokens: {search_results['snippet_tokens']:,}")
    logging.info(f"Total tokens: {search_results['total_tokens']:,}")
    
    # Check if any snippet is stale and show the warning message
    stale_snippets_found = False
    for match in filtered_results:
        try:
            db_file_mod_time = match.payload.get("file_mod_time", 0.0)
            file_path = match.payload.get("file_path", "")
            if file_path and os.path.exists(file_path):
                disk_mod_time = os.path.getmtime(file_path)
                # Use a small tolerance (0.001 seconds) to avoid floating point precision issues
                if disk_mod_time - db_file_mod_time > 0.001:
                    stale_snippets_found = True
                    break
        except Exception:
            pass
    
    # Show warning if any stale snippets were found
    if stale_snippets_found:
        logging.warning("=" * 80)
        logging.warning("STALE ENTRIES FOUND - UPDATE EMBEDDINGS FOR BEST RESULTS")
        logging.warning("RUN: cq embed --recreate -r -d .")
        logging.warning("=" * 80)
