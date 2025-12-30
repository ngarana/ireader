#!/bin/bash
echo "Starting Docker Container..."
docker start ollama-intel

echo "Launching Ollama Server (this window must stay open)..."
docker exec -it ollama-intel bash -c "cd /llm/scripts/ && source ipex-llm-init -g --device iGPU && ./start-ollama.sh"
