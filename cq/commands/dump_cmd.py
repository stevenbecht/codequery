import os
import logging
import argparse
import glob
from pathlib import Path

def register_subparser(subparsers):
    """
    Register the 'dump' subcommand and its arguments.
    """
    dump_parser = subparsers.add_parser("dump", help="Dump contents of files for consumption.")
    
    dump_parser.add_argument(
        "paths", 
        nargs="+", 
        help="Files or directories to dump"
    )
    dump_parser.add_argument(
        "-r", "--recursive", 
        action="store_true",
        help="Recursively process directories"
    )
    dump_parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    dump_parser.add_argument(
        "--include-binary",
        action="store_true",
        help="Include binary files (default: skip binary files)"
    )
    
    dump_parser.set_defaults(func=handle_dump)

def get_files_from_path(path, recursive=False):
    """
    Get all files from a path. If path is a directory and recursive is True,
    recursively get all files from subdirectories.
    """
    path = Path(path)
    
    # Handle direct file paths
    if path.is_file():
        return [str(path)]
    
    # Handle directory paths
    if path.is_dir():
        if recursive:
            return [str(p) for p in path.glob('**/*') if p.is_file()]
        else:
            return [str(p) for p in path.glob('*') if p.is_file()]
    
    # Handle glob patterns
    if '*' in str(path):
        if recursive:
            # If the pattern is a simple wildcard like *.py, and recursive is True,
            # we want to find all matching files in all subdirectories
            if str(path).startswith('*') or '/' not in str(path):
                # Create a pattern that matches in all subdirectories
                base_dir = '.'
                pattern = str(path)
                # Use **/ pattern for recursive glob
                recursive_pattern = os.path.join('**', pattern)
                return glob.glob(recursive_pattern, recursive=True)
            else:
                # For patterns with directories, use as is with recursive=True
                return glob.glob(str(path), recursive=True)
        else:
            # Non-recursive glob
            return glob.glob(str(path))
    
    # If we get here, the path wasn't found
    logging.warning(f"[Dump] No files found matching: {path}")
    return []

def is_binary_file(file_path):
    """
    Check if a file is binary by reading a small chunk and looking for null bytes.
    """
    try:
        with open(file_path, 'rb') as file:
            chunk = file.read(4096)
            return b'\0' in chunk or b'\xff' in chunk
    except Exception:
        return True  # Assume binary if we can't read it
    
def is_excluded_file(file_path):
    """
    Check if a file should be excluded (like __pycache__ directories).
    """
    parts = Path(file_path).parts
    return "__pycache__" in parts or ".git" in parts

def dump_file(file_path, include_binary=False):
    """
    Dump the contents of a file in the BEGIN/END format.
    """
    if not os.path.isfile(file_path):
        logging.warning(f"[Dump] Not a file: {file_path}")
        return
    
    if is_excluded_file(file_path):
        if logging.getLogger().level <= logging.DEBUG:
            logging.debug(f"[Dump] Skipping excluded file: {file_path}")
        return
    
    if not include_binary and is_binary_file(file_path):
        if logging.getLogger().level <= logging.DEBUG:
            logging.debug(f"[Dump] Skipping binary file: {file_path}")
        return
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        if include_binary:
            logging.warning(f"[Dump] Unable to read {file_path} as UTF-8 text, even though --include-binary was set.")
        else:
            logging.debug(f"[Dump] Skipping binary file: {file_path}")
        return
    except Exception as e:
        logging.warning(f"[Dump] Error reading {file_path}: {e}")
        return
    
    # Use relative path for output if possible
    rel_path = os.path.relpath(file_path)
    
    print(f"\nBEGIN: {rel_path}")
    print(content, end="" if content.endswith("\n") else "\n")
    print(f"END: {rel_path}")

def handle_dump(args):
    """
    Handle the dump subcommand.
    """
    # Process each path argument
    all_files = []
    
    for path in args.paths:
        files = get_files_from_path(path, args.recursive)
        all_files.extend(files)
    
    if not all_files:
        logging.warning("[Dump] No files found matching the specified paths.")
        return
    
    # Filter files
    filtered_files = []
    for file in all_files:
        if not is_excluded_file(file) and (args.include_binary or not is_binary_file(file)):
            filtered_files.append(file)
    
    if args.verbose:
        logging.info(f"[Dump] Found {len(filtered_files)} files to dump (out of {len(all_files)} total files)")
    
    if not filtered_files:
        logging.warning("[Dump] No text files found after filtering.")
        return
    
    # Sort files for consistent output
    filtered_files.sort()
    
    # Dump each file
    for file in filtered_files:
        dump_file(file, args.include_binary) 