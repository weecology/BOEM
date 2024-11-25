#!/bin/bash

# Check if required argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <git_branch>"
    exit 1
fi

GIT_BRANCH=$1

# Set the container name for Serenity
CONTAINER_NAME="serenity.azurecr.io/boem:latest"

# Pull the latest BOEM container from Serenity
echo "Pulling the latest BOEM container from Serenity..."
docker pull $CONTAINER_NAME

# Run the container, clone the repo, checkout the specified branch, and run the code
docker run -it --rm $CONTAINER_NAME /bin/bash -c "
    git clone https://github.com/your-repo-url.git /app && \
    cd /app && \
    git checkout $GIT_BRANCH && \
    pip install -r requirements.txt && \
    python main.py
"
