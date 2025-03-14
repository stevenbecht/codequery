# ./cq/cli.py

import argparse
import sys
import logging

# We import our command modules:
from cq.commands import embed_cmd
from cq.commands import search_cmd
from cq.commands import chat_cmd
from cq.commands import stats_cmd
from cq.commands import diff_cmd
from cq.commands import db_cmd

def setup_logging(verbose: bool):
    """
    Removes all existing handlers, then attaches exactly one handler
    so we fully control which logs appear. Blocks "HTTP Request:" lines
    if not verbose. 
    """
    root_logger = logging.getLogger()
    # 1) Remove any existing handlers
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    # 2) Decide overall log level
    level = logging.DEBUG if verbose else logging.INFO
    root_logger.setLevel(level)

    # 3) Create a single stream handler with your desired format
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    handler.setLevel(level)

    # 4) If not verbose, attach a filter that hides lines containing "HTTP Request:"
    if not verbose:
        class HideHttpRequestsFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                if "HTTP Request:" in record.getMessage():
                    return False
                return True

        handler.addFilter(HideHttpRequestsFilter())

    # 5) Attach your single handler
    root_logger.addHandler(handler)

def main():
    parser = argparse.ArgumentParser(
        prog="cq",
        description="Code-Query CLI with subcommands"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register subcommands from each module:
    embed_cmd.register_subparser(subparsers)
    search_cmd.register_subparser(subparsers)
    chat_cmd.register_subparser(subparsers)
    stats_cmd.register_subparser(subparsers)
    diff_cmd.register_subparser(subparsers)
    db_cmd.register_subparser(subparsers)

    # Parse CLI args, set up logging, and dispatch:
    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))
    args.func(args)
