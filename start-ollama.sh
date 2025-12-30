#!/bin/bash
#!/bin/bash
echo "[*] Ensuring Docker Container is UP..."
docker start ollama-intel

echo "[*] Launching Ollama Server..."
echo "    (Wait for 'Listening on [::]:11434' before running your python script)"
echo "    (Press Ctrl+C to stop)"

# We use a Heredoc (<<EOF) to pass commands cleanly into the container
docker exec -it ollama-intel bash <<EOF
    cd /llm/scripts/
    source ipex-llm-init -g --device iGPU
    # If the script exists, run it. Otherwise try standard serve.
    if [ -f "./start-ollama.sh" ]; then
        ./start-ollama.sh
    else
        echo "Script not found, trying 'ollama serve'..."
        ollama serve
    fi
EOF
