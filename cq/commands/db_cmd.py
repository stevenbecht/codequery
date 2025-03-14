import argparse
import subprocess
import sys
import os
import logging

def get_script_path():
    """
    Find the absolute path to the run_qdrant.sh script.
    
    The script is located at the root of the project.
    """
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels (from cq/commands to project root)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    # Construct the path to the script
    script_path = os.path.join(project_root, 'run_qdrant.sh')
    
    if not os.path.exists(script_path):
        logging.error(f"Could not find run_qdrant.sh at {script_path}")
        sys.exit(1)
        
    return script_path

def run_qdrant_command(command):
    """
    Run the run_qdrant.sh script with the specified command.
    """
    script_path = get_script_path()
    
    try:
        # Make sure the script is executable
        if not os.access(script_path, os.X_OK):
            subprocess.run(['chmod', '+x', script_path], check=True)
        
        # Run the script with the specified command
        result = subprocess.run(
            [script_path, command], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Print the output
        print(result.stdout.strip())
        
        return 0
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            logging.error(e.stderr)
        return 1

def db_command(args):
    """
    Handle the db command and dispatch to the appropriate action.
    """
    return run_qdrant_command(args.action)

def register_subparser(subparsers):
    """
    Register the 'db' subcommand and its arguments.
    """
    db_parser = subparsers.add_parser(
        "db", 
        help="Manage the Qdrant database container."
    )
    
    db_parser.add_argument(
        "action",
        choices=["start", "stop", "restart", "status"],
        help="Action to perform on the Qdrant container"
    )
    
    db_parser.set_defaults(func=db_command) 