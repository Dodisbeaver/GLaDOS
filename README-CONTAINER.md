# GLaDOS Containerized Deployment

This document describes the fully containerized GLaDOS deployment that **eliminates the need for Python on the host system**.

## Architecture Overview

```
Browser ←→ Audio Proxy ←→ GLaDOS Container
   ↑         (Node.js)        (Python)
WebRTC      Port 3000      Port 8765
```

### Services
- **Audio Proxy**: WebRTC ↔ WebSocket bridge (Node.js)
- **GLaDOS Container**: Core voice processing (Python + models)
- **TTS API**: Optional standalone TTS service

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Ollama running (can be containerized separately)
- **Optional**: Create `ollama-network` if connecting to external Ollama:
  ```bash
  docker network create ollama-network
  ```
  *Note: This is only needed if you want GLaDOS to communicate with an Ollama instance running in a separate Docker network. For host-based Ollama (via `host.docker.internal`), this network is not required.*

### 1. Configuration
```bash
# Copy and customize environment
cp .env.example .env

# Edit .env for your setup
# - Set OLLAMA_BASE_URL for your Ollama instance
# - Choose GLADOS_RUNTIME (cpu or cuda)
```

### 2. Start Services
```bash
# Build and start all services
docker compose up -d

# Check status
docker compose ps
```

### 3. Access GLaDOS
- Open browser to `http://localhost:3000`
- Click "Start Listening" and grant microphone permissions
- Talk to GLaDOS through the web interface!

## Service Details

### GLaDOS Container
- **Image**: Built from `Dockerfile.assistant`
- **Runtime**: CPU or CUDA (via build arg)
- **Audio**: WebRTC via WebSocket (no system audio dependencies)
- **Models**: Downloaded to persistent volume
- **Config**: Mounted from `./configs/`

### Audio Proxy
- **Image**: Node.js Alpine
- **Purpose**: WebRTC ↔ WebSocket bridge
- **Port**: 3000 (web interface)
- **Health**: `/health` endpoint

### TTS API (Optional)
- **Image**: Original TTS service
- **Purpose**: Standalone text-to-speech API
- **Port**: 5050

## Fixed Issues from Original Plan

### ❌ Original Plan Flaws:
1. **Audio Complexity**: Assumed simple device mounting would work
2. **Model Download**: Inefficient build-time downloading
3. **Network Issues**: Wrong Ollama URLs for different platforms
4. **GPU Handling**: No proper ONNX runtime selection

### ✅ WebRTC Solution:
1. **Browser Audio**: No container audio drivers needed
2. **Volume Models**: Persistent storage with runtime download
3. **Smart Networking**: Auto-detects Docker environment
4. **Build Args**: Proper CPU/GPU variant selection

## Troubleshooting

### GLaDOS Container Won't Start
```bash
# Check logs
docker compose logs glados

# Common issues:
# - Ollama unreachable: Check OLLAMA_BASE_URL
# - Models failing: Check volume permissions
```

### Audio Proxy Connection Issues
```bash
# Check proxy logs
docker compose logs audio-proxy

# Verify WebSocket connection
curl http://localhost:3000/health
```

### Ollama Connection
```bash
# Test from container
docker compose exec glados curl $OLLAMA_BASE_URL/api/tags

# Common URLs:
# Docker Desktop: http://host.docker.internal:11434
# Linux: http://localhost:11434 (with network_mode: host)
```

## Development

### Building Images
```bash
# Build specific service
docker compose build glados

# Build with different runtime
GLADOS_RUNTIME=cuda docker compose build glados
```

### Custom Configuration
```bash
# Mount custom config
docker compose run glados uv run glados start --config /app/configs/custom_config.yaml
```

## Production Deployment

### Security Considerations
- Use reverse proxy with SSL
- Firewall audio proxy port appropriately
- Consider authentication for web interface

### Performance
- Use CUDA runtime for GPU acceleration
- Mount models volume on fast storage
- Consider dedicated Ollama server

## Comparison: Before vs After

| Aspect | Original (Host Python) | Containerized (WebRTC) |
|--------|----------------------|----------------------|
| Host Dependencies | Python 3.12, uv, PortAudio | Docker only |
| Audio Handling | Direct system audio | WebRTC via browser |
| Model Storage | Host filesystem | Docker volume |
| Platform Support | OS-specific audio setup | Universal (web browser) |
| Isolation | Shared host Python | Complete containerization |
| Deployment | Manual dependency mgmt | `docker compose up` |

This solution achieves the goal of **complete containerization** while maintaining full GLaDOS functionality!