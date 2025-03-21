import os
import logging
import argparse
import glob
from pathlib import Path
import pathspec

# Import token counting functionality
from cq.embedding import count_tokens

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
        "-c", "--count-tokens",
        action="store_true",
        help="Count tokens in each file and show total"
    )
    dump_parser.add_argument(
        "-l", "--list-tokens",
        action="store_true",
        help="When used with -c, list token count for each file"
    )
    dump_parser.add_argument(
        "--include-binary",
        action="store_true",
        help="Include binary files (default: skip binary files)"
    )
    dump_parser.add_argument(
        "--ignore-gitignore",
        action="store_true",
        help="Ignore .gitignore patterns (default: respect .gitignore)"
    )
    
    dump_parser.set_defaults(func=handle_dump)

def get_gitignore_spec():
    """
    Parse .gitignore file and return a pathspec object to match paths.
    Returns None if no .gitignore file exists or it cannot be read.
    """
    gitignore_path = os.path.join(os.getcwd(), '.gitignore')
    
    if not os.path.isfile(gitignore_path):
        return None
    
    try:
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
        
        return pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern, 
            gitignore_content.splitlines()
        )
    except Exception as e:
        logging.warning(f"[Dump] Error reading .gitignore: {e}")
        return None

def get_files_from_path(path, recursive=False, gitignore_spec=None):
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
            all_files = [str(p) for p in path.glob('**/*') if p.is_file()]
        else:
            all_files = [str(p) for p in path.glob('*') if p.is_file()]
            
        # Filter out files that match .gitignore patterns
        if gitignore_spec:
            base_dir = os.path.abspath(path)
            filtered_files = []
            for file_path in all_files:
                # Get the relative path from the base directory
                rel_path = os.path.relpath(file_path, base_dir)
                if not gitignore_spec.match_file(rel_path):
                    filtered_files.append(file_path)
            return filtered_files
        else:
            return all_files
    
    # Handle glob patterns
    if '*' in str(path):
        if recursive:
            # For simple patterns like "*.ts" or extension-based patterns
            path_str = str(path)
            if path_str.startswith('*') or '/' not in path_str:
                # First, find files in the current directory matching the pattern
                files = glob.glob(path_str)
                
                # Then, search recursively in all subdirectories
                # We need to scan all directories and find matching files
                for root, dirs, _ in os.walk('.'):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        # Apply the pattern in each subdirectory
                        subdir_matches = glob.glob(os.path.join(dir_path, path_str))
                        files.extend(subdir_matches)
                
                # Filter out files that match .gitignore patterns
                if gitignore_spec:
                    files = [f for f in files if not gitignore_spec.match_file(f)]
                
                return files
            else:
                # For patterns with directories, use as is with recursive=True
                files = glob.glob(str(path), recursive=True)
                
                # Filter out files that match .gitignore patterns
                if gitignore_spec:
                    files = [f for f in files if not gitignore_spec.match_file(f)]
                
                return files
        else:
            # Non-recursive glob
            files = glob.glob(str(path))
            
            # Filter out files that match .gitignore patterns
            if gitignore_spec:
                files = [f for f in files if not gitignore_spec.match_file(f)]
            
            return files
    
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

def dump_file(file_path, include_binary=False, count_tokens_flag=False, list_tokens_flag=False, print_content=True):
    """
    Dump the contents of a file in the BEGIN/END format.
    Returns token count if count_tokens_flag is True, otherwise returns 0.
    """
    if not os.path.isfile(file_path):
        logging.warning(f"[Dump] Not a file: {file_path}")
        return 0
    
    if is_excluded_file(file_path):
        if logging.getLogger().level <= logging.DEBUG:
            logging.debug(f"[Dump] Skipping excluded file: {file_path}")
        return 0
    
    if not include_binary and is_binary_file(file_path):
        if logging.getLogger().level <= logging.DEBUG:
            logging.debug(f"[Dump] Skipping binary file: {file_path}")
        return 0
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        if include_binary:
            logging.warning(f"[Dump] Unable to read {file_path} as UTF-8 text, even though --include-binary was set.")
        else:
            logging.debug(f"[Dump] Skipping binary file: {file_path}")
        return 0
    except Exception as e:
        logging.warning(f"[Dump] Error reading {file_path}: {e}")
        return 0
    
    # Calculate the relative path from the current directory
    # Make sure we use the absolute path first to handle different path formats
    abs_path = os.path.abspath(file_path)
    current_dir = os.path.abspath(os.getcwd())
    
    # If the file is outside the current directory, use the full path
    if os.path.commonpath([current_dir, abs_path]) != current_dir:
        display_path = file_path
    else:
        # Otherwise, get the relative path from the current directory
        display_path = os.path.relpath(abs_path, current_dir)
        
        # Add './' prefix for files in the current directory (not in subdirectories)
        if '/' not in display_path and '\\' not in display_path:
            display_path = './' + display_path
    
    # Ensure forward slashes for consistency across platforms
    display_path = display_path.replace(os.sep, '/')
    
    # Count tokens if requested
    token_count = 0
    if count_tokens_flag:
        token_count = count_tokens(content)
        
        # If both count and list flags are set, print token count for this file
        if list_tokens_flag:
            logging.info(f"Tokens: {token_count:,} | File: {display_path}")
    
    # Print file content only if requested
    if print_content:
        print(f"\nBEGIN: {display_path}")
        print(content, end="" if content.endswith("\n") else "\n")
        print(f"END: {display_path}")
    
    return token_count

def handle_dump(args):
    """
    Handle the dump subcommand.
    """
    # Parse .gitignore if it exists and we're not ignoring it
    gitignore_spec = None
    if not args.ignore_gitignore:
        gitignore_spec = get_gitignore_spec()
        if gitignore_spec and args.verbose:
            logging.info("[Dump] Using .gitignore patterns to filter files")
    
    # Process each path argument
    all_files = []
    
    for path in args.paths:
        files = get_files_from_path(path, args.recursive, gitignore_spec)
        if args.verbose and files:
            logging.info(f"[Dump] Found {len(files)} files matching: {path}")
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
    
    # Determine if we should print file contents
    # If -v is specified, always print content regardless of other flags
    # Otherwise, don't print content if -c is specified
    print_content = args.verbose or not args.count_tokens
    
    # For token counting with file listing, print a header
    if args.count_tokens and args.list_tokens:
        logging.info("=== Token Count Per File ===")
    
    # Dump each file and accumulate token counts if requested
    total_tokens = 0
    for file in filtered_files:
        file_tokens = dump_file(
            file, 
            args.include_binary, 
            args.count_tokens, 
            args.list_tokens,
            print_content
        )
        total_tokens += file_tokens
    
    # Print total token count if counting was enabled
    if args.count_tokens:
        if args.list_tokens:
            logging.info(f"=== Total Tokens: {total_tokens:,} ===")
        else:
            logging.info(f"Total Tokens: {total_tokens:,}") 