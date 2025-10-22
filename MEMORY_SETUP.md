# GLaDOS Memory System Setup Guide

## Overview

GLaDOS now has a working RAG (Retrieval-Augmented Generation) memory system that:
- ✅ Uses **cosine similarity** for accurate retrieval
- ✅ Retrieves relevant context **before** storing new memories
- ✅ Supports bulk ingestion from **markdown files**

## Quick Start

### 1. Add Your Household Knowledge

Create markdown files in the `knowledge/` directory:

```bash
knowledge/
├── household.md          # Family info, schedules, preferences
├── devices.md            # Smart home device names and locations
├── procedures.md         # How-to instructions
└── contacts.md           # Emergency contacts, service providers
```

See `knowledge/example_household.md` for a template.

### 2. Ingest Your Knowledge (Inside Docker)

```bash
# Start GLaDOS containers
docker-compose up -d

# Run ingestion script inside the glados container
docker-compose exec glados python scripts/ingest_markdown_memories.py \
  --input-dir /app/knowledge \
  --config /app/configs/glados_config.yaml

# Or dry-run to preview first
docker-compose exec glados python scripts/ingest_markdown_memories.py \
  --input-dir /app/knowledge \
  --dry-run
```

### 3. Configure Memory Settings

Edit `configs/glados_config.yaml`:

```yaml
memory:
  enable_memory: true
  memory_path: /app/data/memory

  # Embedding provider options:
  # - "sentence_transformers" (CPU, fast, good quality)
  # - "ollama" (requires Ollama server with embeddinggemma)
  embedding_provider: sentence_transformers
  embedding_model: all-MiniLM-L6-v2

  # Retrieval settings
  max_retrievals: 5              # Max memories per query
  similarity_threshold: 0.7      # Lower = more results (0.5-0.7 recommended)

  # Storage
  store_user_inputs: true        # Store user messages as episodic memories
  auto_select_provider: false    # Auto-select best embedding model for hardware
```

### 4. Verify It Works

Ask GLaDOS questions about the knowledge you ingested:
- "What time is dinner?"
- "Who is allergic to peanuts?"
- "What's the thermostat set to at night?"

GLaDOS should retrieve relevant facts and include them in responses.

## How It Works

### Memory Flow

1. **User speaks**: "What time is dinner?"
2. **Retrieve memories**: Search vector DB with cosine similarity
3. **Format context**: "Previous relevant context: Dinner is usually at 6:00 PM"
4. **LLM generates response**: Includes retrieved facts in the prompt
5. **Store conversation**: User input saved for future retrieval

### Memory Types

- **`semantic`**: General knowledge, facts (e.g., "Dinner at 6 PM")
- **`episodic`**: Specific conversations (e.g., "User asked about dinner yesterday")
- **`procedural`**: How-to knowledge (e.g., "To brew coffee, first...")

## Embedding Models

### Sentence Transformers (CPU, Recommended)

Fast, runs anywhere, good quality:

```yaml
embedding_provider: sentence_transformers
embedding_model: all-MiniLM-L6-v2  # 384 dim, fast
# OR
embedding_model: all-mpnet-base-v2  # 768 dim, slower but better
```

### Ollama with EmbeddingGemma (GPU)

Requires Ollama server running:

```yaml
embedding_provider: ollama
embedding_model: embeddinggemma
ollama_base_url: http://ollama:11434
```

**⚠️ Important**: If you change embedding models, you must:
1. Clear existing memories: `memory.clear_memory()`
2. Re-ingest all knowledge with the new model

## Troubleshooting

### No Memories Retrieved?

**Problem**: Retrieval returns empty results

**Solutions**:
1. Lower similarity threshold in config:
   ```yaml
   similarity_threshold: 0.5  # or even 0.3 for testing
   ```

2. Verify memories were stored:
   ```bash
   docker-compose exec glados python -c "
   from glados.core.memory_core import MemoryCore
   m = MemoryCore(memory_path='/app/data/memory', enable_memory=True)
   print(m.get_memory_stats())
   "
   ```

3. Test retrieval directly:
   ```python
   from glados.core.memory_core import MemoryCore
   memory = MemoryCore(memory_path='/app/data/memory', enable_memory=True)

   results = memory.retrieve_memories("dinner time")
   for r in results:
       print(f"[{r['similarity']:.3f}] {r['text'][:100]}")
   ```

### Wrong Results?

**Problem**: Retrieval finds irrelevant memories

**Solutions**:
1. Increase similarity threshold (more strict):
   ```yaml
   similarity_threshold: 0.7  # or 0.8
   ```

2. Improve markdown structure:
   - Use clear headings
   - Break up large sections
   - Include synonyms and variations

3. Reduce chunk size:
   ```bash
   --max-chunk-size 300  # Smaller = more focused chunks
   ```

### Cosine Similarity Issues?

**Verify the fix is applied**:

```bash
grep -n 'metric("cosine")' src/glados/core/memory_core.py
```

Should show:
```
245:            search = self.table.search(query_embedding).metric("cosine").limit(max_results)
```

## Advanced Usage

### Custom Metadata

Add custom metadata to track source, timestamp, etc.:

```python
memory.store_memory(
    text="John likes coffee",
    memory_type="semantic",
    speaker="system",
    metadata={
        "source": "household.md",
        "category": "preferences",
        "confidence": 1.0
    }
)
```

### Filter by Memory Type

```python
# Only search episodic memories (conversations)
results = memory.retrieve_memories(
    query="What did we discuss yesterday?",
    memory_types=["episodic"]
)

# Only search semantic facts
results = memory.retrieve_memories(
    query="What's the wifi password?",
    memory_types=["semantic"]
)
```

### Exclude Recent Memories

```python
# Get recently stored IDs
recent = memory.get_recent_memories(hours=1)
recent_ids = [m['id'] for m in recent]

# Retrieve while excluding recent
results = memory.retrieve_memories(
    query="Tell me about the house",
    exclude_ids=recent_ids
)
```

## Performance Tips

1. **Chunk Size**: 300-400 words is optimal for most use cases
2. **Max Retrievals**: 5 is a good default; more = longer prompts
3. **Similarity Threshold**: 0.7 for strict, 0.5 for lenient
4. **GPU Acceleration**: Use Ollama + embeddinggemma if you have GPU

## Testing

Run the demo to verify everything works:

```bash
python demo_markdown_ingestion.py
```

This will:
- Load `knowledge/example_household.md`
- Chunk by headings
- Embed with sentence-transformers
- Store in temp LanceDB
- Test retrieval with sample queries

## Files Reference

- `src/glados/core/memory_core.py` - Core memory system (FIXED: cosine similarity)
- `src/glados/core/llm_processor.py` - LLM processor (FIXED: retrieve before store)
- `scripts/ingest_markdown_memories.py` - Bulk ingestion script
- `knowledge/` - Your markdown knowledge base
- `knowledge/README.md` - Detailed ingestion guide
- `demo_markdown_ingestion.py` - Standalone demo/test
- `test_memory_fix.py` - Verification test for cosine fix

## Next Steps

1. ✅ Create your household markdown files in `knowledge/`
2. ✅ Run ingestion script inside Docker
3. ✅ Adjust similarity threshold if needed
4. ✅ Test retrieval with voice commands
5. ✅ Monitor logs for retrieval quality

See `knowledge/README.md` for detailed markdown formatting tips!
