#!/bin/bash

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
