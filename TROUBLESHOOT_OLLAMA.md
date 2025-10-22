# Troubleshooting Ollama Connection

## The Error You Saw

```
Failed to connect to Ollama server at http://localhost:11434
```

## Root Cause

The config had `memory_ollama_base_url` but the code expects `memory_ollama_url`.

## ✅ Fixed!

Changed in `configs/glados_config.yaml`:

```yaml
# WRONG (old):
memory_ollama_base_url: "http://host.docker.internal:11434"

# CORRECT (fixed):
memory_ollama_url: "http://host.docker.internal:11434"
```

## Quick Verification Steps

### 1. Check Your Laptop's Ollama is Running

```bash
# On your laptop (not Docker):
curl http://localhost:11434/api/tags

# Should return:
# {"models":[{"name":"embeddinggemma",...}]}
```

If this fails, start Ollama:
```bash
ollama serve
```

### 2. Find Your Connection Method

#### Option A: Docker Desktop (Mac/Windows) - `host.docker.internal`

```yaml
# In configs/glados_config.yaml:
memory_ollama_url: "http://host.docker.internal:11434"
```

Test from container:
```bash
docker-compose exec glados curl http://host.docker.internal:11434/api/tags
```

#### Option B: Linux or Custom Setup - Use Laptop IP

Find your IP:
```bash
# Mac/Linux:
ifconfig | grep "inet " | grep -v 127.0.0.1

# Windows:
ipconfig

# Example output: 192.168.1.100
```

Update config:
```yaml
memory_ollama_url: "http://192.168.1.100:11434"
```

Test from container:
```bash
docker-compose exec glados curl http://192.168.1.100:11434/api/tags
```

### 3. Restart GLaDOS Container

After fixing the config:

```bash
docker-compose restart glados
```

### 4. Check the Logs

```bash
docker-compose logs glados | grep -i "ollama\|embedding"
```

**Success looks like:**
```
✅ Ollama provider ready: embeddinggemma (dim=768)
✅ Memory Core initialized successfully
```

**Failure looks like:**
```
❌ Failed to connect to Ollama server at http://localhost:11434
❌ Failed to initialize Ollama provider
```

## Common Issues

### Issue 1: "Connection Refused"

**Symptom:**
```
Connection refused [Errno 111]
```

**Cause:** Ollama not running or wrong URL

**Fix:**
1. Start Ollama on your laptop: `ollama serve`
2. Verify it's accessible: `curl http://localhost:11434/api/tags`
3. Check firewall isn't blocking port 11434

### Issue 2: "host.docker.internal" Not Working

**Symptom:**
```
Failed to establish a new connection
```

**Cause:** `host.docker.internal` doesn't work on Linux Docker

**Fix:** Use your laptop's actual IP address:
```yaml
memory_ollama_url: "http://192.168.1.100:11434"  # Your IP
```

### Issue 3: "Model Not Found"

**Symptom:**
```
Model embeddinggemma not available
```

**Cause:** Model not pulled

**Fix:**
```bash
# On your laptop:
ollama pull embeddinggemma
ollama list  # Verify it shows
```

### Issue 4: Ollama Listening on Wrong Interface

**Symptom:** Works from laptop but not from Docker

**Cause:** Ollama only listening on 127.0.0.1

**Fix:** Make Ollama listen on all interfaces:

```bash
# On your laptop, set environment variable:
export OLLAMA_HOST=0.0.0.0:11434

# Then start Ollama:
ollama serve
```

Or add to your shell profile:
```bash
# ~/.bashrc or ~/.zshrc
export OLLAMA_HOST=0.0.0.0:11434
```

## Testing Connection Manually

### From Your Laptop

```bash
# Test Ollama API
curl http://localhost:11434/api/tags

# Test embedding generation
curl http://localhost:11434/api/embeddings -d '{
  "model": "embeddinggemma",
  "prompt": "test"
}'
```

### From Docker Container

```bash
# Test with host.docker.internal
docker-compose exec glados curl http://host.docker.internal:11434/api/tags

# Test with IP (replace with yours)
docker-compose exec glados curl http://192.168.1.100:11434/api/tags

# Test embedding from container
docker-compose exec glados curl http://host.docker.internal:11434/api/embeddings -d '{
  "model": "embeddinggemma",
  "prompt": "test"
}'
```

### From Python in Container

```bash
docker-compose exec glados python -c "
import requests
url = 'http://host.docker.internal:11434/api/tags'
try:
    response = requests.get(url, timeout=5)
    print(f'✅ Connected! Status: {response.status_code}')
    print(f'Models: {response.json()}')
except Exception as e:
    print(f'❌ Failed: {e}')
"
```

## Network Diagram

```
┌─────────────────────┐
│  Your Laptop        │
│  - Ollama Server    │
│  - embeddinggemma   │
│  Port: 11434        │
│  IP: 192.168.1.100  │ ← Find this with ifconfig
└──────────┬──────────┘
           │
           │ Network Access Required
           │
┌──────────▼──────────┐
│  Docker Container   │
│  - GLaDOS           │
│  - Tries to connect │
│    via config URL   │
└─────────────────────┘

Connection URLs that work:
✅ http://host.docker.internal:11434  (Mac/Windows Docker Desktop)
✅ http://192.168.1.100:11434         (Your laptop IP)
❌ http://localhost:11434             (Wrong - refers to container!)
❌ http://127.0.0.1:11434             (Wrong - refers to container!)
```

## Verification Checklist

Before starting GLaDOS, verify:

- [ ] Ollama is running on your laptop: `curl http://localhost:11434/api/tags`
- [ ] embeddinggemma is pulled: `ollama list | grep embeddinggemma`
- [ ] Config has correct key: `memory_ollama_url` (not `memory_ollama_base_url`)
- [ ] URL is accessible from container: `docker-compose exec glados curl <URL>/api/tags`
- [ ] Firewall allows port 11434 (if using IP address)

## Still Not Working?

### Fallback: Use Sentence Transformers

If you can't get Ollama working, use CPU-based sentence transformers temporarily:

```yaml
# In configs/glados_config.yaml:
memory_embedding_provider: "sentence_transformers"
memory_embedding_model: "all-MiniLM-L6-v2"
# memory_ollama_url: "..."  # Comment out or remove
```

This runs embeddings inside the container (slower but works everywhere).

### Debug Mode

Enable detailed logging:

```bash
# Add to docker-compose.yml under glados service:
environment:
  - LOG_LEVEL=DEBUG

# Restart and watch logs
docker-compose restart glados
docker-compose logs -f glados
```

## Quick Fix Summary

1. ✅ Fix config key: `memory_ollama_url` (not `memory_ollama_base_url`)
2. ✅ Use correct URL: `http://host.docker.internal:11434` or your laptop IP
3. ✅ Ensure Ollama is running: `ollama serve`
4. ✅ Test connection from container: `docker-compose exec glados curl <URL>/api/tags`
5. ✅ Restart GLaDOS: `docker-compose restart glados`

Your config is now fixed - just restart the container!
