#!/bin/bash

### Create Network

NETWORK="genai_chat_net"
if docker network inspect ${NETWORK} > /dev/null 2>&1
then
    echo "Network ${NETWORK} already exists"
else
    echo "Network ${NETWORK} doesn't exist, creating it"
    docker network create ${NETWORK}
fi

### Build the Docker images

echo "Building Docker image: nginx-base"
docker build -t nginx-base -f nginx/Dockerfile .

echo "Building Docker image: genai-chat-backend"
docker build -t genai-chat-backend -f backend/Dockerfile .

if [ $? -eq 0 ]; then
   echo Images Built
else
   echo Images not Build
   exit 1
fi

### Deploy the containers
echo "Deploying the containers"

# Prefer Docker Compose v2 (`docker compose`); fall back to the legacy
# standalone `docker-compose` binary for older Docker installations.
if docker compose version > /dev/null 2>&1; then
    COMPOSE="docker compose"
elif command -v docker-compose > /dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    echo "Error: neither 'docker compose' (v2) nor 'docker-compose' (v1) is available."
    echo "Install Docker Desktop or the docker-compose-plugin and re-run."
    exit 1
fi

echo "Using: ${COMPOSE}"
${COMPOSE} up -d