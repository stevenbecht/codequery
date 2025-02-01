import argparse
import sys
import logging

# We import our command modules:
from cq.commands import embed_cmd
from cq.commands import search_cmd
from cq.commands import chat_cmd
from cq.commands import stats_cmd
from cq.commands import diff_cmd

def setup_logging(verbose: bool):
    """
    Sets up the root logger level to DEBUG if verbose is True,
    otherwise INFO. Outputs to stdout instead of stderr.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,  # Use stdout explicitly
    )

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

    # Parse CLI args, set up logging, and dispatch:
    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))
    args.func(args)
