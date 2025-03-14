#!/bin/bash

# Function to start the container
start_container() {
    # Check if the container 'cq-qdrant' exists
    if docker inspect cq-qdrant >/dev/null 2>&1; then
        # If it exists, check if it's already running
        if [ "$(docker inspect -f '{{.State.Running}}' cq-qdrant)" = "true" ]; then
            echo "Container cq-qdrant is already running."
        else
            echo "Starting existing container cq-qdrant..."
            docker start cq-qdrant
        fi
    else
        echo "Running new container cq-qdrant..."
        docker run -d \
          -p 6333:6333 \
          -v qdrant_storage:/qdrant/storage \
          --name cq-qdrant \
          qdrant/qdrant
    fi
}

# Function to stop the container
stop_container() {
    # Check if the container exists
    if docker inspect cq-qdrant >/dev/null 2>&1; then
        # Check if it's running
        if [ "$(docker inspect -f '{{.State.Running}}' cq-qdrant)" = "true" ]; then
            echo "Stopping container cq-qdrant..."
            docker stop cq-qdrant
        else
            echo "Container cq-qdrant is already stopped."
        fi
    else
        echo "Container cq-qdrant does not exist."
    fi
}

# Function to restart the container
restart_container() {
    stop_container
    sleep 2  # Give it a moment to fully stop
    start_container
}

# Function to check container status
status_container() {
    # Check if the container exists
    if docker inspect cq-qdrant >/dev/null 2>&1; then
        # Check if it's running
        if [ "$(docker inspect -f '{{.State.Running}}' cq-qdrant)" = "true" ]; then
            echo "Container cq-qdrant is running."
            echo "Container info:"
            docker ps -f name=cq-qdrant --format "ID: {{ .ID }}\nCreated: {{ .CreatedAt }}\nStatus: {{ .Status }}\nPorts: {{ .Ports }}"
        else
            echo "Container cq-qdrant exists but is not running."
        fi
    else
        echo "Container cq-qdrant does not exist."
    fi
}

# Display usage information
usage() {
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  start    - Start the Qdrant container (default if no command specified)"
    echo "  stop     - Stop the Qdrant container"
    echo "  restart  - Restart the Qdrant container"
    echo "  status   - Show the status of the Qdrant container"
    echo "  help     - Display this help message"
}

# Main script logic
case "${1:-start}" in
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        restart_container
        ;;
    status)
        status_container
        ;;
    help)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac
