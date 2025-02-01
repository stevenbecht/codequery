import sys
import logging
import os
import xml.etree.ElementTree as ET

def register_subparser(subparsers):
    """
    Register the 'diff' subcommand and its arguments.
    """
    diff_parser = subparsers.add_parser("diff", help="Check or apply code diffs from an XML file.")
    diff_parser.add_argument(
        "--check-diff", action="store_true",
        help="Check the proposed diff for validity."
    )
    diff_parser.add_argument(
        "--apply-diff", action="store_true",
        help="Apply the proposed diff to the codebase."
    )
    diff_parser.add_argument(
        "--diff-file", required=True,
        help="Path to the XML diff file."
    )
    diff_parser.add_argument(
        "--codebase-dir", required=True,
        help="Path to the codebase on disk to apply changes."
    )
    diff_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output for diff. Shows lines replaced if --apply-diff or --check-diff."
    )
    diff_parser.set_defaults(func=handle_diff)

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
    """
    logging.info(f"[Diff] Reading diff file: {args.diff_file}")
    if not os.path.isfile(args.diff_file):
        logging.error(f"Diff file not found: {args.diff_file}")
        sys.exit(1)

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
                    f"[Diff] Invalid line range {start}-{end} for file {file_path} "
                    f"(file length={len(original_lines)})."
                )
                if args.check_diff:
                    continue
                else:
                    sys.exit(1)

            new_code_lines = ch["new_code"].split("\n")

            if args.check_diff:
                logging.info(f"[Diff] Check only: validated changes for {file_path}")
                if args.verbose:
                    logging.info(f"[Diff][Verbose] Proposed replacement in {file_path}, lines {start}-{end}:")
                    for line in new_code_lines:
                        logging.info(f"    > {line}")
            elif args.apply_diff:
                if args.verbose:
                    logging.info(f"[Diff][Verbose] Replacing lines {start}-{end} in {file_path} with:")
                    for line in new_code_lines:
                        logging.info(f"    > {line}")

                original_lines[start:end+1] = [line + "\n" for line in new_code_lines]

        if args.apply_diff and not args.check_diff:
            with open(full_path, "w", encoding="utf-8") as f:
                f.writelines(original_lines)
            logging.info(f"[Diff] Applied changes to {file_path} successfully.")
