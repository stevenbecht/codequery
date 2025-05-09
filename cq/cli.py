# ./cq/cli.py

import argparse
import sys
import logging
import openai

# We import our command modules:
from cq.commands import embed_cmd
from cq.commands import search_cmd
from cq.commands import chat_cmd
from cq.commands import stats_cmd
from cq.commands import diff_cmd
from cq.commands import db_cmd
from cq.commands import dump_cmd

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
    
    # Create a custom formatter that supports the 'end' parameter
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            # Get the standard formatted message
            message = super().format(record)
            
            # Check if the record has an 'end' attribute
            if hasattr(record, 'end'):
                # If end is empty, remove the newline
                if record.end == "":
                    return message.rstrip('\n')
            
            return message
    
    handler.setFormatter(CustomFormatter("%(levelname)s: %(message)s"))
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
    dump_cmd.register_subparser(subparsers)

    # Parse CLI args, set up logging, and dispatch:
    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))
    
    try:
        args.func(args)
    except openai.BadRequestError as e:
        logging.error(f"OpenAI API Error: {e}")
        error_msg = str(e)
        if "maximum context length" in error_msg and "tokens" in error_msg:
            logging.error("\nSuggestions:")
            logging.error("1. Use -n/--no-context flag to chat without code context")
            logging.error("2. Use a smaller directory with 'cq dump <smaller_dir>'") 
            logging.error("3. Use a model with larger context (e.g. --model 'openai:gpt-4-turbo')")
            logging.error("4. Use -w/--max-window to limit context tokens (e.g. -w 4000)")
        sys.exit(1)
    except openai.AuthenticationError:
        logging.error("Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable.")
        sys.exit(1)
    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        logging.error(f"Connection error: {e}")
        logging.error("Please check your internet connection and ensure required services are running.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback
            logging.error(traceback.format_exc())
        sys.exit(1)
