# Knowledge Base for GLaDOS Memory

This directory contains markdown files that can be bulk-ingested into GLaDOS's memory system.

## How It Works

1. **Write markdown files** with your household information, facts, preferences, etc.
2. **Run the ingestion script** to load them into the vector database
3. **GLaDOS retrieves relevant facts** during conversations using RAG

## Quick Start

### 1. Prepare Your Markdown Files

Create `.md` files in this directory (or subdirectories) with structured information:

```markdown
# Category Name

## Subsection

Content here will be chunked and stored as semantic memories.

Each heading section becomes a searchable chunk.
```

### 2. Preview What Will Be Ingested (Dry Run)

```bash
python scripts/ingest_markdown_memories.py \
  --input-dir knowledge \
  --dry-run
```

### 3. Ingest Into Memory

Using default sentence-transformers:
```bash
python scripts/ingest_markdown_memories.py \
  --input-dir knowledge \
  --memory-path data/memory
```

Using your GLaDOS config (recommended):
```bash
python scripts/ingest_markdown_memories.py \
  --input-dir knowledge \
  --config configs/glados_config.yaml
```

### 4. Verify Ingestion

Check memory stats:
```bash
python -c "
from src.glados.core.memory_core import MemoryCore
import sys
sys.path.insert(0, 'src')

memory = MemoryCore(memory_path='data/memory', enable_memory=True)
print(memory.get_memory_stats())
"
```

## Memory Types

The script supports three memory types:

- **`semantic`** (default): General knowledge, facts, reference information
  - Use for: household info, preferences, device names, schedules

- **`episodic`**: Specific events, conversations, experiences
  - Use for: "We discussed this on Tuesday", "Last time you asked..."

- **`procedural`**: How-to knowledge, procedures, recipes
  - Use for: "To turn on the lights, say...", "Coffee brewing steps..."

Example:
```bash
python scripts/ingest_markdown_memories.py \
  --input-dir knowledge/procedures \
  --memory-type procedural \
  --speaker system
```

## Advanced Options

### Chunk Size

Control how large each memory chunk is (in words):
```bash
--max-chunk-size 300  # Smaller chunks = more precise retrieval
--max-chunk-size 600  # Larger chunks = more context per result
```

**Recommendation**: 300-400 words for factual data, 200-300 for dense information

### Similarity Threshold

Edit your `configs/glados_config.yaml`:
```yaml
memory:
  similarity_threshold: 0.7  # Default (strict matching)
  similarity_threshold: 0.5  # More lenient (finds more results)
  similarity_threshold: 0.3  # Very lenient (may include less relevant)
```

## Important Notes

### ⚠️ Changing Embedding Models

If you change your embedding provider/model, you **must re-ingest**:

```bash
# Clear existing memories
python -c "
from src.glados.core.memory_core import MemoryCore
memory = MemoryCore(memory_path='data/memory', enable_memory=True)
memory.clear_memory()
print('✅ Memory cleared')
"

# Re-ingest with new model
python scripts/ingest_markdown_memories.py \
  --input-dir knowledge \
  --config configs/glados_config.yaml
```

### 📊 Recommended Embedding Models

- **CPU / Default**: `sentence_transformers` with `all-MiniLM-L6-v2` (384 dim)
  - Fast, good quality, runs anywhere

- **Better Quality (CPU)**: `sentence_transformers` with `all-mpnet-base-v2` (768 dim)
  - Slower but more accurate

- **GPU / Ollama**: `embeddinggemma` via Ollama (768 dim)
  - Requires Ollama server running
  - Update config: `embedding_provider: ollama`, `embedding_model: embeddinggemma`

## Example Files

See `example_household.md` for a template showing:
- Family member info
- Daily routines
- Home automation device names
- Dietary preferences
- Emergency contacts

## Tips for Good Retrieval

1. **Use clear headings** - Each heading becomes searchable context
2. **Write naturally** - Phrase things as you'd ask them
3. **Include synonyms** - "Coffee maker" and "coffee machine"
4. **Avoid huge walls of text** - Break into logical sections
5. **Test retrieval** - Ask GLaDOS questions to verify it finds the right info

## Testing Retrieval

After ingesting, test if GLaDOS can find your information:

```python
from src.glados.core.memory_core import MemoryCore

memory = MemoryCore(memory_path='data/memory', enable_memory=True)

# Test query
results = memory.retrieve_memories("What time is dinner?")
for mem in results:
    print(f"[{mem['similarity']:.3f}] {mem['text'][:100]}")
```

Expected: Should find "Dinner is usually at 6:00 PM" with high similarity.

## Memory Maintenance

### Deduplication

Over time, similar conversations can create duplicate memories. Clean them up using:

```bash
# Preview what would be removed (dry run)
python scripts/cleanup_memory.py --dry-run

# Remove duplicates (90% similarity threshold)
python scripts/cleanup_memory.py

# More aggressive deduplication (85% similarity)
python scripts/cleanup_memory.py --threshold 0.85

# Keep oldest memories instead of newest
python scripts/cleanup_memory.py --strategy keep_oldest

# Show statistics only
python scripts/cleanup_memory.py --stats-only
```

**Deduplication Strategies:**
- `keep_newest` (default): Retains most recent duplicate
- `keep_oldest`: Retains earliest duplicate
- `keep_longest`: Retains duplicate with most content

**Recommended Schedule:**
- Run weekly for active conversations
- Run monthly for occasional use
- Always use `--dry-run` first to preview changes

## Troubleshooting

**No results returned?**
- Check similarity threshold (lower it to 0.5 or 0.3)
- Verify memories were ingested: `memory.get_memory_stats()`
- Test with very similar wording to what's in the markdown

**Wrong results?**
- Chunk size may be too large - try `--max-chunk-size 300`
- Add more context to your markdown sections
- Use more specific headings

**Slow retrieval?**
- LanceDB cosine search is fast, but embedding the query takes time
- Consider GPU acceleration or faster embedding model
- Reduce `max_retrievals` in config (default is 5)

**Too many duplicate memories?**
- Run `python scripts/cleanup_memory.py --dry-run` to see duplicates
- Use deduplication to clean up (see Memory Maintenance above)
- Adjust the similarity threshold in `store_memory()` calls
