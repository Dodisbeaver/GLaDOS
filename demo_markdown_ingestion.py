#!/usr/bin/env python3
"""
Standalone demo of markdown ingestion without requiring full glados imports.

This demonstrates the chunking and ingestion flow for household knowledge.
"""

import tempfile
from pathlib import Path
from sentence_transformers import SentenceTransformer
import lancedb


def chunk_markdown(content: str, max_chunk_size: int = 400) -> list[dict[str, str]]:
    """Split markdown by headings."""
    chunks = []
    lines = content.split('\n')
    current_heading = None
    current_section = []

    for line in lines:
        if line.startswith('#'):
            # Save previous section
            if current_section:
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    chunks.append({
                        'text': section_text,
                        'heading': current_heading or 'Introduction'
                    })

            current_heading = line.lstrip('#').strip()
            current_section = [line]
        else:
            current_section.append(line)

    # Last section
    if current_section:
        section_text = '\n'.join(current_section).strip()
        if section_text:
            chunks.append({
                'text': section_text,
                'heading': current_heading or 'Introduction'
            })

    return chunks


def demo_ingestion():
    """Demonstrate markdown ingestion with actual embedding and storage."""
    print("=" * 70)
    print("Markdown Knowledge Base Ingestion Demo")
    print("=" * 70)

    # Load example file
    knowledge_file = Path("knowledge/example_household.md")

    if not knowledge_file.exists():
        print(f"❌ Example file not found: {knowledge_file}")
        return False

    print(f"\n📄 Reading: {knowledge_file}")
    content = knowledge_file.read_text()

    # Chunk the content
    print("✂️  Chunking markdown by headings...")
    chunks = chunk_markdown(content, max_chunk_size=400)
    print(f"✅ Created {len(chunks)} chunks")

    # Show chunks
    print("\n" + "=" * 70)
    print("Chunks Preview")
    print("=" * 70)

    for i, chunk in enumerate(chunks[:5], 1):  # Show first 5
        print(f"\nChunk {i}: {chunk['heading']}")
        print("-" * 70)
        preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
        print(preview)

    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")

    # Initialize embedding model
    print("\n" + "=" * 70)
    print("Embedding and Storing")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("🔧 Loading embedding model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_dim = model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded (dimension: {embedding_dim})")

        # Create vector database
        print(f"💾 Creating LanceDB in: {tmpdir}")
        db = lancedb.connect(tmpdir)

        # Embed and store all chunks
        print("🔄 Embedding and storing chunks...")
        data = []

        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk['text'])
            data.append({
                'id': f"household_{i}",
                'text': chunk['text'],
                'embedding': embedding.tolist(),
                'heading': chunk['heading'],
                'source': 'example_household.md',
                'memory_type': 'semantic',
                'speaker': 'system'
            })

        table = db.create_table("memories", data)
        print(f"✅ Stored {len(data)} memories in vector database")

        # Test retrieval
        print("\n" + "=" * 70)
        print("Testing Retrieval")
        print("=" * 70)

        test_queries = [
            "What time is dinner?",
            "Who is allergic to peanuts?",
            "What temperature should the thermostat be at night?",
            "What is John's favorite drink?",
        ]

        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            query_embedding = model.encode(query)

            # Use cosine similarity (the fix we applied!)
            results = table.search(query_embedding).metric("cosine").limit(2).to_list()

            for j, result in enumerate(results, 1):
                similarity = 1.0 - result['_distance']
                print(f"  {j}. [{similarity:.3f}] {result['heading']}")

                # Show relevant excerpt
                text = result['text']
                # Find the most relevant sentence
                sentences = text.split('.')
                for sentence in sentences:
                    if any(word.lower() in sentence.lower() for word in query.split()):
                        print(f"     → {sentence.strip()}.")
                        break

        print("\n" + "=" * 70)
        print("✅ Demo Complete!")
        print("=" * 70)
        print("\nThis demonstrates:")
        print("  1. ✅ Markdown chunking by headings")
        print("  2. ✅ Embedding with sentence-transformers")
        print("  3. ✅ Storage in LanceDB with cosine metric")
        print("  4. ✅ Semantic retrieval of household facts")
        print("\nTo ingest your own files into GLaDOS:")
        print("  → Use scripts/ingest_markdown_memories.py inside Docker")
        print("  → See knowledge/README.md for full instructions")

    return True


if __name__ == "__main__":
    try:
        demo_ingestion()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
