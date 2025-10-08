#!/bin/bash
set -e

# GLaDOS Container Entrypoint - Fixes model download and config issues

echo "Starting GLaDOS container..."

# Check if models directory is populated
if [ ! -f "${GLADOS_MODELS_PATH}/silero_vad.onnx" ]; then
    echo "Downloading GLaDOS models..."
    uv run glados download
    echo "Models downloaded successfully"
else
    echo "Models already present, skipping download"
fi

# Check if custom config is mounted, otherwise use default
if [ ! -f "${GLADOS_CONFIG_PATH}" ]; then
    echo "Using default GLaDOS configuration"
    export GLADOS_CONFIG_PATH="/app/configs/glados_config.yaml"
fi

# Update config for containerized environment
if [ "${GLADOS_AUDIO_IO}" = "webrtc" ]; then
    echo "Configuring WebRTC audio mode..."
    # This would ideally update the config file or use env vars
    # For now, we'll assume the config supports env var overrides
    export GLADOS_AUDIO_IO_TYPE=webrtc
    export GLADOS_WEBSOCKET_PORT=8765
fi

# Ensure Ollama URL is accessible from container
if [ -z "${OLLAMA_BASE_URL}" ]; then
    # Try to detect if we're in Docker Desktop vs Linux
    if [ -f "/.dockerenv" ]; then
        if getent hosts host.docker.internal > /dev/null 2>&1; then
            export OLLAMA_BASE_URL="http://host.docker.internal:11434"
            echo "Using Docker Desktop Ollama URL: ${OLLAMA_BASE_URL}"
        else
            # Linux - assume host network or external service
            export OLLAMA_BASE_URL="http://localhost:11434"
            echo "Using Linux host network Ollama URL: ${OLLAMA_BASE_URL}"
        fi
    fi
fi

echo "Starting GLaDOS with config: ${GLADOS_CONFIG_PATH}"
echo "Ollama URL: ${OLLAMA_BASE_URL}"
echo "Audio I/O: ${GLADOS_AUDIO_IO_TYPE:-sounddevice}"

# Execute the command
exec "$@"