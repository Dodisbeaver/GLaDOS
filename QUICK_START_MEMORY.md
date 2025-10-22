# GLaDOS Memory System - Quick Start

## ✅ What You Get

- **Persistent storage** on your laptop (not Docker volumes)
- **Your laptop's Ollama** for embeddings (embeddinggemma)
- **Bulk markdown ingestion** for household knowledge
- **Ready for web UI** (read-write mounts)

## 🚀 5-Minute Setup

### 1. Update Config (Choose Your Laptop IP)

Edit `configs/glados_config.yaml`:

```yaml
memory_embedding_provider: "ollama"
memory_embedding_model: "embeddinggemma"

# Mac/Windows Docker Desktop:
memory_ollama_url: "http://host.docker.internal:11434"

# OR Linux / if above fails (use YOUR laptop IP):
memory_ollama_url: "http://192.168.1.100:11434"
```

### 2. Ensure Ollama Has embeddinggemma

On your **laptop** (not Docker):

```bash
ollama pull embeddinggemma
ollama list  # Should show embeddinggemma
```

### 3. Add Your Knowledge

Create `knowledge/household.md`:

```markdown
# Family Info

John likes coffee in the morning.
Sarah prefers tea.
Emma is allergic to peanuts.

## Schedule

Dinner is at 6:00 PM.
Kids bedtime is 8:30 PM.
```

### 4. Start & Ingest

```bash
# Start GLaDOS
docker-compose up -d

# Ingest your knowledge
docker-compose exec glados python scripts/ingest_markdown_memories.py \
  --input-dir /app/knowledge \
  --config /app/configs/glados_config.yaml
```

### 5. Verify It Works

```bash
# Check memory is stored on host
ls -la data/memory/

# Should show:
# memories.lance/  ← Your vector database!
```

Ask GLaDOS:
- "What time is dinner?"
- "Who is allergic to peanuts?"

## 📂 File Structure

```
your-project/
├── docker-compose.yml          # ✅ Updated with host mounts
├── data/
│   └── memory/                 # ← Vector DB (persistent on host)
│       └── memories.lance/
├── knowledge/                  # ← Your markdown files
│   ├── example_household.md
│   └── household.md           # ← Add yours here
├── configs/
│   ├── glados_config.yaml     # ← Main config
│   └── glados_laptop_ollama_config.yaml  # ← Template
└── scripts/
    └── ingest_markdown_memories.py  # ← Bulk ingestion
```

## 🔧 Key Configuration

**docker-compose.yml** (already updated):
```yaml
volumes:
  - ./data/memory:/app/data/memory    # ← Host mount (persistent!)
  - ./knowledge:/app/knowledge        # ← Read-write for web UI
  - ./configs:/app/configs:ro
```

**glados_config.yaml**:
```yaml
memory_enabled: true
memory_embedding_provider: "ollama"
memory_embedding_model: "embeddinggemma"
memory_ollama_url: "http://host.docker.internal:11434"
memory_similarity_threshold: 0.6  # Lower = more results
```

## 🎯 How It Works

```
1. You ask: "What time is dinner?"
   ↓
2. GLaDOS embeds your question using laptop's Ollama
   ↓
3. LanceDB searches ./data/memory/ with cosine similarity
   ↓
4. Finds: "Dinner is at 6:00 PM" (similarity: 0.78)
   ↓
5. GLaDOS responds with context from memory
```

## 📝 Adding More Knowledge

### Via Files (Now)

```bash
# 1. Create markdown in knowledge/
echo "# New Info\nWifi password is ABC123" > knowledge/wifi.md

# 2. Re-ingest
docker-compose exec glados python scripts/ingest_markdown_memories.py \
  --input-dir /app/knowledge \
  --config /app/configs/glados_config.yaml
```

### Via Web UI (Future)

The setup is **ready for web UI**:
- Upload endpoint writes to `/app/knowledge/` → `./knowledge/` on host
- Trigger re-ingestion from UI
- Files persist on your laptop
- No Docker volume complexity!

## 🔍 Testing & Verification

### Test Ollama Connection

```bash
# From container to your laptop
docker-compose exec glados curl http://host.docker.internal:11434/api/tags

# Should return: {"models":[{"name":"embeddinggemma",...}]}
```

### Test Memory Retrieval

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

# Test query
results = memory.retrieve_memories('What time is dinner?')
for r in results:
    print(f'[{r[\"similarity\"]:.3f}] {r[\"text\"][:100]}')
"
```

### Check Logs

```bash
# Watch for memory retrieval
docker-compose logs -f glados | grep -i "retrieved.*memories"

# Should show:
# LLM Processor: Retrieved 3 relevant memories
```

## ⚠️ Important Notes

### Changing Embedding Models

If you change from `embeddinggemma` to another model (or vice versa):

```bash
# 1. Clear existing vectors (incompatible dimensions!)
docker-compose exec glados python -c "
from glados.core.memory_core import MemoryCore
import sys
sys.path.insert(0, '/app/src')
m = MemoryCore(memory_path='/app/data/memory', enable_memory=True)
m.clear_memory()
print('✅ Cleared')
"

# 2. Update config with new provider/model

# 3. Re-ingest everything
docker-compose exec glados python scripts/ingest_markdown_memories.py \
  --input-dir /app/knowledge \
  --config /app/configs/glados_config.yaml
```

### Similarity Threshold

Adjust based on retrieval quality:

```yaml
memory_similarity_threshold: 0.7  # Strict (fewer, better matches)
memory_similarity_threshold: 0.6  # Balanced (recommended for embeddinggemma)
memory_similarity_threshold: 0.5  # Lenient (more matches)
memory_similarity_threshold: 0.3  # Very lenient (for testing)
```

## 📚 Full Documentation

- **`MEMORY_SETUP.md`** - Complete memory system guide
- **`LAPTOP_OLLAMA_SETUP.md`** - Detailed Ollama configuration
- **`knowledge/README.md`** - Markdown formatting tips
- **`CONTRIBUTING.md`** - Development guide

## 🎉 You're Done!

Your memory system is now:
- ✅ Storing vectors on your laptop (`./data/memory/`)
- ✅ Using your laptop's Ollama for embeddings
- ✅ Ready to ingest markdown knowledge
- ✅ Configured for future web UI

**Next**: Add your household knowledge to `knowledge/` and run the ingestion script!
