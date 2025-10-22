# Using Your Laptop's Ollama for GLaDOS Memory Embeddings

## Overview

This guide shows you how to:
1. ✅ Store memory files **persistently on your host** (not Docker volumes)
2. ✅ Use your **laptop's Ollama** server for embeddings (avoiding Docker overhead)
3. ✅ Set up for future **web UI** to add knowledge

## Architecture

```
┌─────────────────────┐
│  Your Laptop        │
│  - Ollama Server    │
│  - embeddinggemma   │
│  Port: 11434        │
└──────────┬──────────┘
           │
           │ Network
           │
┌──────────▼──────────┐
│  Docker Container   │
│  - GLaDOS           │
│  - Memory requests  │
└──────────┬──────────┘
           │
           │ Host mount
           │
┌──────────▼──────────┐
│  Host Filesystem    │
│  ./data/memory/     │ ← Vector DB stored here
│  ./knowledge/       │ ← Markdown files here
└─────────────────────┘
```

## Step 1: Update docker-compose.yml

**Already done!** Your docker-compose.yml now has:

```yaml
volumes:
  - ./data/memory:/app/data/memory          # Host directory ✅
  - ./knowledge:/app/knowledge              # Read-write for web UI ✅
  - ./configs:/app/configs:ro               # Read-only configs ✅
```

This means:
- **`./data/memory/`** on your laptop = persistent LanceDB
- **`./knowledge/`** on your laptop = your markdown files
- Future web UI can write to both from inside the container

## Step 2: Set Up Your Laptop's Ollama

### Install embeddinggemma

On your laptop (not in Docker):

```bash
# Pull the embedding model
ollama pull embeddinggemma

# Verify it works
ollama run embeddinggemma "test"
```

### Find Your Laptop's IP Address

```bash
# On Mac/Linux:
ifconfig | grep "inet " | grep -v 127.0.0.1

# On Windows:
ipconfig

# Example output: 192.168.1.100
```

### Test Ollama API

```bash
# Test from your laptop (should work)
curl http://localhost:11434/api/tags

# Test from Docker network (use your laptop IP)
curl http://192.168.1.100:11434/api/tags
```

## Step 3: Configure GLaDOS to Use Your Laptop's Ollama

### Option A: Use the Pre-configured File

Copy the template:

```bash
cp configs/glados_laptop_ollama_config.yaml configs/glados_config.yaml
```

Then edit `configs/glados_config.yaml` and update line 20:

```yaml
# If on Mac/Windows with Docker Desktop:
memory_ollama_url: "http://host.docker.internal:11434"

# OR if that doesn't work, use your laptop's IP:
memory_ollama_url: "http://192.168.1.100:11434"  # ← YOUR IP HERE
```

### Option B: Use Environment Variable

Create or edit `.env` file in your project root:

```bash
# .env file
GLADOS_MEMORY_OLLAMA_URL=http://host.docker.internal:11434
# OR
GLADOS_MEMORY_OLLAMA_URL=http://192.168.1.100:11434
```

Then in your config, use:

```yaml
memory_ollama_url: "${GLADOS_MEMORY_OLLAMA_URL}"
```

## Step 4: Create Your Knowledge Base

Add markdown files to `knowledge/`:

```bash
knowledge/
├── household.md        # Family info, routines
├── devices.md          # Smart home devices
├── preferences.md      # Dietary, entertainment
└── contacts.md         # Emergency contacts
```

See `knowledge/example_household.md` for a template.

## Step 5: Start GLaDOS and Ingest Knowledge

### Start the containers:

```bash
docker-compose up -d
```

### Check the logs to verify Ollama connection:

```bash
docker-compose logs glados | grep -i "ollama\|embedding"
```

You should see:
```
✅ Ollama provider ready: embeddinggemma (dim=768)
```

### Ingest your markdown files:

```bash
docker-compose exec glados python scripts/ingest_markdown_memories.py \
  --input-dir /app/knowledge \
  --config /app/configs/glados_config.yaml
```

### Verify ingestion:

```bash
docker-compose exec glados python -c "
from glados.core.memory_core import MemoryCore
import sys
sys.path.insert(0, '/app/src')

memory = MemoryCore(
    memory_path='/app/data/memory',
    embedding_provider='ollama',
    embedding_model='embeddinggemma',
    ollama_url='http://host.docker.internal:11434',
    enable_memory=True
)

stats = memory.get_memory_stats()
print(f'Total memories: {stats[\"total_memories\"]}')
print(f'By type: {stats[\"by_type\"]}')
print(f'Model: {stats[\"embedding_model\"]}')
"
```

## Step 6: Verify Memory Storage on Host

Check that files are actually on your laptop:

```bash
ls -lh data/memory/
```

You should see LanceDB files:
```
data/memory/
├── memories.lance/
│   ├── data/
│   ├── _versions/
│   └── _latest.manifest
```

These files are now **persisted on your host**, not in Docker!

## Step 7: Test Retrieval

Ask GLaDOS questions and verify it retrieves from memory:

```bash
# Check logs for memory retrieval
docker-compose logs -f glados | grep -i "retrieved.*memories"
```

You should see:
```
LLM Processor: Retrieved 3 relevant memories
```

## Configuration Reference

### Full Config for Laptop Ollama

```yaml
Glados:
  # Memory settings
  memory_enabled: true
  memory_path: "/app/data/memory"

  # Use Ollama on your laptop
  memory_embedding_provider: "ollama"
  memory_embedding_model: "embeddinggemma"
  memory_ollama_url: "http://host.docker.internal:11434"

  # Retrieval settings
  memory_max_retrievals: 5
  memory_similarity_threshold: 0.6  # Lower for embeddinggemma (better quality)
  memory_store_assistant_responses: true
  memory_store_user_inputs: true

  # Optional: Don't auto-select since we're explicit
  memory_auto_select_provider: false
```

### Environment Variables

You can override via `.env`:

```bash
# Ollama URL for embeddings
GLADOS_MEMORY_OLLAMA_URL=http://192.168.1.100:11434

# Enable/disable memory
GLADOS_MEMORY_ENABLED=true

# Memory path (inside container)
GLADOS_MEMORY_PATH=/app/data/memory
```

## Troubleshooting

### Can't Connect to Laptop Ollama

**Problem**: `Failed to connect to Ollama server`

**Solutions**:

1. **Try `host.docker.internal`** (Mac/Windows Docker Desktop):
   ```yaml
   memory_ollama_url: "http://host.docker.internal:11434"
   ```

2. **Use laptop's actual IP** (Linux or if above fails):
   ```bash
   # Find IP
   ifconfig | grep "inet "

   # Use in config
   memory_ollama_url: "http://192.168.1.100:11434"
   ```

3. **Check firewall** - Make sure port 11434 is accessible:
   ```bash
   # From inside container
   docker-compose exec glados curl http://host.docker.internal:11434/api/tags
   ```

4. **Ollama listen address** - Make sure Ollama listens on all interfaces:
   ```bash
   # On your laptop, set environment variable
   export OLLAMA_HOST=0.0.0.0:11434

   # Then restart Ollama
   ollama serve
   ```

### Memory Files Not on Host

**Problem**: Data disappears after container restart

**Check**:
```bash
# Should show host mount
docker-compose exec glados ls -la /app/data/memory

# Should show files on host
ls -la ./data/memory/
```

**Fix**: Make sure docker-compose.yml has:
```yaml
- ./data/memory:/app/data/memory  # NOT glados-memory volume!
```

### Web UI Future Feature

When you build the web UI for adding knowledge:

1. **Upload endpoint** saves to `/app/knowledge/` (mapped to `./knowledge/`)
2. **Re-ingest button** calls the ingestion script
3. **Files persist** on host filesystem
4. **No Docker volume issues** - everything is on the host

Example structure:
```python
# Future web UI endpoint
@app.post("/upload-knowledge")
async def upload_knowledge(file: UploadFile):
    # Save to /app/knowledge/ (host mounted)
    filepath = Path("/app/knowledge") / file.filename
    filepath.write_bytes(await file.read())

    # Trigger re-ingestion
    ingest_markdown_memories(
        input_dir="/app/knowledge",
        memory_path="/app/data/memory",
        config_path="/app/configs/glados_config.yaml"
    )

    return {"status": "ingested", "file": file.filename}
```

## Performance Notes

### Why Use Laptop's Ollama?

1. **✅ Better GPU utilization** - Your laptop GPU isn't shared with container
2. **✅ Faster cold starts** - Model stays loaded in laptop memory
3. **✅ Less container bloat** - No need to mount huge model files
4. **✅ Reuse existing setup** - Same Ollama for LLM + embeddings

### Embedding Speed

With embeddinggemma (768 dim):
- **Single query**: ~50-200ms (depending on GPU)
- **Bulk ingestion**: ~100-500 chunks/minute
- **Retrieval**: ~50-100ms (embedding query + LanceDB search)

For 100 markdown chunks (typical household knowledge):
- **Initial ingestion**: ~2-5 minutes
- **After that**: Instant (vectors are cached)

### Similarity Threshold Tuning

embeddinggemma produces higher quality embeddings than sentence-transformers:

```yaml
# Sentence transformers (384 dim):
memory_similarity_threshold: 0.7  # Need higher threshold

# EmbeddingGemma (768 dim):
memory_similarity_threshold: 0.6  # Can use lower threshold for same quality
```

Test and adjust based on retrieval quality!

## Quick Command Reference

```bash
# Ingest markdown
docker-compose exec glados python scripts/ingest_markdown_memories.py \
  --input-dir /app/knowledge --config /app/configs/glados_config.yaml

# Check memory stats
docker-compose exec glados python -c "from glados.core.memory_core import MemoryCore; import sys; sys.path.insert(0, '/app/src'); m = MemoryCore(memory_path='/app/data/memory', enable_memory=True); print(m.get_memory_stats())"

# Clear all memories (if changing embedding model)
docker-compose exec glados python -c "from glados.core.memory_core import MemoryCore; import sys; sys.path.insert(0, '/app/src'); m = MemoryCore(memory_path='/app/data/memory', enable_memory=True); m.clear_memory(); print('Cleared')"

# Test Ollama connection from container
docker-compose exec glados curl http://host.docker.internal:11434/api/tags

# View GLaDOS logs
docker-compose logs -f glados
```

## Next Steps

1. ✅ Start Ollama on your laptop with embeddinggemma
2. ✅ Update `memory_ollama_url` in config
3. ✅ Create markdown files in `knowledge/`
4. ✅ Run ingestion script
5. ✅ Verify files appear in `./data/memory/` on host
6. ✅ Test GLaDOS retrieval with questions
7. ✅ Build web UI when ready (files already set up for it!)

All your memory data will be safely stored on your laptop's filesystem! 🎉
