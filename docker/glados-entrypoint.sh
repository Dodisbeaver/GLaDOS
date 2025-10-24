#!/bin/bash
set -e

# GLaDOS Container Entrypoint - Fixes model download and config issues

echo "Starting GLaDOS container..."

# Fix permissions for mounted volumes if running as root
# (This handles the case where host is root and container user is non-root)
if [ "$(id -u)" = "0" ]; then
    echo "Running as root, fixing volume permissions..."
    chown -R glados:glados /app/models /app/data 2>/dev/null || true
    echo "Permissions fixed, will drop to glados user after setup"
elif [ ! -w "/app/models" ]; then
    echo "WARNING: /app/models is not writable by current user"
    echo "You may need to fix permissions on the host with:"
    echo "  chown -R 1000:1000 ./models ./data"
fi

# Check if models directory is populated
if [ ! -f "${GLADOS_MODELS_PATH}/ASR/silero_vad_v5.onnx" ]; then
    echo "Downloading GLaDOS models..."
    uv run glados download
    echo "Models downloaded successfully"
else
    echo "Models already present, skipping download"
fi

# Copy Piper models if they don't exist in the volume
if [ ! -f "${GLADOS_MODELS_PATH}/TTS/glados_piper_medium.onnx" ]; then
    echo "Copying Piper TTS models to volume..."
    mkdir -p "${GLADOS_MODELS_PATH}/TTS"
    if [ -f "/app/src_models/TTS/glados_piper_medium.onnx" ]; then
        cp /app/src_models/TTS/glados_piper_medium.onnx* "${GLADOS_MODELS_PATH}/TTS/"
        echo "Piper models copied successfully"
    else
        echo "Warning: Piper models not found in build, will be downloaded by glados download if available"
    fi
else
    echo "Piper models already present"
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

# Execute the command as glados user if we're root
if [ "$(id -u)" = "0" ]; then
    echo "Dropping privileges to glados user..."
    exec gosu glados "$@"
else
    exec "$@"
fi