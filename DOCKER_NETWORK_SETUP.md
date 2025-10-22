# Docker Network Setup for Ollama

## Overview

Your Ollama is running in its own Docker network (`ollama-network`), and GLaDOS needs to connect to it to use embeddings.

## ✅ What Was Changed

### Updated `docker-compose.yml`

**Added to glados service:**
```yaml
networks:
  - default            # GLaDOS internal network
  - ollama-network     # ← Connect to Ollama's network
```

**Added to networks section:**
```yaml
networks:
  default:
    name: glados-network
  ollama-network:
    external: true     # ← Use existing network
```

## Verification Steps

### 1. Check Ollama Network Exists

```bash
docker network ls | grep ollama
```

Should show:
```
<network-id>   ollama-network   bridge   local
```

### 2. Verify Ollama Container is on That Network

```bash
docker network inspect ollama-network
```

Should show your Ollama container in the output. Look for:
```json
"Containers": {
    "<container-id>": {
        "Name": "ollama",
        ...
    }
}
```

### 3. Restart GLaDOS to Join the Network

```bash
docker-compose down
docker-compose up -d
```

### 4. Verify GLaDOS is on Both Networks

```bash
docker inspect glados-glados-1 | grep -A 10 "Networks"
```

Should show both:
- `glados-network` (default)
- `ollama-network` (for Ollama access)

### 5. Test Connection from GLaDOS Container

```bash
docker-compose exec glados curl http://ollama:11434/api/tags
```

Should return:
```json
{"models":[{"name":"embeddinggemma",...}]}
```

### 6. Check GLaDOS Logs

```bash
docker-compose logs glados | grep -i "ollama\|embedding"
```

Should show:
```
✅ Ollama provider ready: embeddinggemma (dim=768)
✅ Memory Core initialized successfully
```

## Network Diagram

```
┌─────────────────────────────────┐
│  ollama-network                 │
│  (External Docker Network)      │
│                                 │
│  ┌──────────────┐              │
│  │  Ollama      │              │
│  │  Port: 11434 │              │
│  └──────────────┘              │
│         ▲                       │
└─────────┼───────────────────────┘
          │
          │ Network bridge
          │
┌─────────┼───────────────────────┐
│         │                       │
│  ┌──────┴───────┐              │
│  │  GLaDOS      │              │
│  │  (joined)    │              │
│  └──────────────┘              │
│                                 │
│  ┌──────────────┐              │
│  │  audio-proxy │              │
│  └──────────────┘              │
│                                 │
│  glados-network                 │
│  (Default GLaDOS Network)       │
└─────────────────────────────────┘
```

## Configuration Summary

Your config now has:
```yaml
# configs/glados_config.yaml
memory_ollama_url: "http://ollama:11434"
```

This works because:
1. ✅ GLaDOS container is on `ollama-network`
2. ✅ Ollama container is on `ollama-network`
3. ✅ Docker DNS resolves `ollama` to the container name

## Common Issues

### "Network ollama-network not found"

**Symptom:**
```
ERROR: Network ollama-network declared as external, but could not be found
```

**Fix:** The network doesn't exist yet. Create it or check the name:

```bash
# List all networks
docker network ls

# If it's named differently, update docker-compose.yml
# OR create the network:
docker network create ollama-network
```

### Ollama Container Not on ollama-network

**Symptom:** Connection still fails after restart

**Fix:** Add Ollama to the network:

```bash
# Find Ollama container name
docker ps | grep ollama

# Connect it to the network
docker network connect ollama-network <ollama-container-name>
```

### Multiple Networks Confusion

If your Ollama is on a different network name:

```bash
# Find which network Ollama is on
docker inspect <ollama-container-name> | grep -A 10 "Networks"
```

Then update `docker-compose.yml` with the correct network name:
```yaml
networks:
  default:
    name: glados-network
  your-actual-ollama-network-name:  # ← Update this
    external: true
```

## Alternative: All Services on One Network

If you prefer, you can run everything on the same network:

**Option 1: Put GLaDOS on Ollama's network only**
```yaml
# docker-compose.yml - glados service
networks:
  - ollama-network  # Only use Ollama network

# At bottom:
networks:
  ollama-network:
    external: true
```

**Option 2: Connect Ollama to GLaDOS network**
```bash
docker network connect glados-network <ollama-container-name>
```

Then use:
```yaml
# configs/glados_config.yaml
memory_ollama_url: "http://ollama:11434"
```

## Verification Commands Reference

```bash
# 1. Check networks exist
docker network ls

# 2. See what's on ollama-network
docker network inspect ollama-network

# 3. See GLaDOS container networks
docker inspect glados-glados-1 | grep -A 10 "Networks"

# 4. Test connection from GLaDOS
docker-compose exec glados curl http://ollama:11434/api/tags

# 5. Test embedding
docker-compose exec glados curl http://ollama:11434/api/embeddings -d '{
  "model": "embeddinggemma",
  "prompt": "test"
}'

# 6. Check logs
docker-compose logs glados | tail -50
```

## Quick Start Checklist

- [ ] Confirm ollama-network exists: `docker network ls`
- [ ] Confirm Ollama is on that network: `docker network inspect ollama-network`
- [ ] Updated docker-compose.yml with external network ✅ (already done)
- [ ] Updated config with `memory_ollama_url: "http://ollama:11434"` ✅ (already done)
- [ ] Restart GLaDOS: `docker-compose down && docker-compose up -d`
- [ ] Test connection: `docker-compose exec glados curl http://ollama:11434/api/tags`
- [ ] Check logs for success: `docker-compose logs glados | grep "Ollama provider ready"`

Once you see "Ollama provider ready", you're all set! 🎉
