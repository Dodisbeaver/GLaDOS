#!/usr/bin/env python3
"""
Bulk ingestion script for loading markdown files into GLaDOS memory.

This script:
1. Reads markdown files from a specified directory
2. Chunks them by headings or paragraph breaks
3. Stores each chunk as a semantic memory
4. Preserves source file metadata for traceability

Usage:
    python scripts/ingest_markdown_memories.py --input-dir /path/to/markdown --config configs/glados_config.yaml
    python scripts/ingest_markdown_memories.py --input-dir ./knowledge --max-chunk-size 300
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import - if running in Docker it will work, otherwise give helpful error
try:
    from glados.core.memory_core import MemoryCore
    from loguru import logger
except ImportError as e:
    print("❌ Import Error: This script should be run inside the Docker container")
    print("\nTo run inside Docker:")
    print("  docker-compose exec glados python scripts/ingest_markdown_memories.py --help")
    print("\nOr install dependencies:")
    print("  pip install pyyaml loguru lancedb sentence-transformers")
    sys.exit(1)


def chunk_markdown(content: str, max_chunk_size: int = 400) -> list[dict[str, str]]:
    """
    Split markdown content into semantic chunks.

    Strategy:
    - Split on headings (# ## ###)
    - Keep heading with its content
    - Further split large sections by paragraphs if needed

    Args:
        content: Raw markdown text
        max_chunk_size: Maximum tokens per chunk (approximate by words)

    Returns:
        List of chunks with metadata
    """
    chunks = []
    lines = content.split('\n')

    current_heading = None
    current_section = []

    for line in lines:
        # Check if this is a heading
        if line.startswith('#'):
            # Save previous section
            if current_section:
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    # Split large sections
                    for chunk in _split_large_section(section_text, max_chunk_size):
                        chunks.append({
                            'text': chunk,
                            'heading': current_heading or 'Introduction'
                        })

            # Start new section
            current_heading = line.lstrip('#').strip()
            current_section = [line]
        else:
            current_section.append(line)

    # Don't forget the last section
    if current_section:
        section_text = '\n'.join(current_section).strip()
        if section_text:
            for chunk in _split_large_section(section_text, max_chunk_size):
                chunks.append({
                    'text': chunk,
                    'heading': current_heading or 'Introduction'
                })

    return chunks


def _split_large_section(text: str, max_size: int) -> list[str]:
    """Split large sections by paragraphs if they exceed max_size words."""
    words = text.split()

    if len(words) <= max_size:
        return [text]

    # Split by paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_words = len(para.split())

        if current_size + para_words > max_size and current_chunk:
            # Save current chunk
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_words
        else:
            current_chunk.append(para)
            current_size += para_words

    # Add remaining
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def load_config(config_path: Optional[str] = None) -> dict:
    """Load memory configuration from YAML config file."""
    if not config_path:
        return {}

    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Extract memory-related config
    return config.get('memory', {})


def ingest_markdown_files(
    input_dir: str,
    memory_path: str = "data/memory",
    config_path: Optional[str] = None,
    max_chunk_size: int = 400,
    memory_type: str = "semantic",
    speaker: str = "system",
    dry_run: bool = False
):
    """
    Ingest all markdown files from a directory into memory.

    Args:
        input_dir: Directory containing .md files
        memory_path: Path to memory database
        config_path: Optional path to GLaDOS config YAML
        max_chunk_size: Maximum words per chunk
        memory_type: Type of memory (semantic, episodic, procedural)
        speaker: Speaker label (system, user, assistant)
        dry_run: If True, only show what would be ingested
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return False

    # Find all markdown files
    md_files = list(input_path.glob("**/*.md"))

    if not md_files:
        logger.warning(f"No markdown files found in {input_dir}")
        return False

    logger.info(f"Found {len(md_files)} markdown file(s)")

    # Load config if provided
    mem_config = load_config(config_path)

    # Initialize MemoryCore
    if not dry_run:
        logger.info("Initializing MemoryCore...")

        memory = MemoryCore(
            memory_path=memory_path,
            embedding_model=mem_config.get('embedding_model', 'all-MiniLM-L6-v2'),
            embedding_provider=mem_config.get('embedding_provider', 'sentence_transformers'),
            max_retrievals=mem_config.get('max_retrievals', 5),
            similarity_threshold=mem_config.get('similarity_threshold', 0.7),
            enable_memory=True,
            auto_select_provider=mem_config.get('auto_select_provider', False),
        )

        if not memory.enable_memory:
            logger.error("Failed to initialize memory core")
            return False

        logger.success(f"MemoryCore initialized with {memory.encoder.get_model_name()}")

    # Process each file
    total_chunks = 0

    for md_file in md_files:
        logger.info(f"Processing: {md_file.name}")

        try:
            content = md_file.read_text(encoding='utf-8')

            # Chunk the content
            chunks = chunk_markdown(content, max_chunk_size)

            logger.info(f"  → Split into {len(chunks)} chunk(s)")

            # Store each chunk
            for i, chunk_data in enumerate(chunks):
                chunk_text = chunk_data['text']
                heading = chunk_data['heading']

                metadata = {
                    'source_file': str(md_file.name),
                    'source_path': str(md_file.relative_to(input_path)),
                    'heading': heading,
                    'chunk_index': i,
                }

                if dry_run:
                    print(f"\n  Chunk {i+1}/{len(chunks)} ({heading}):")
                    print(f"  {'-' * 60}")
                    preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                    print(f"  {preview}")
                    print(f"  Metadata: {metadata}")
                else:
                    mem_id = memory.store_memory(
                        text=chunk_text,
                        memory_type=memory_type,
                        speaker=speaker,
                        metadata=metadata
                    )

                    if mem_id:
                        logger.debug(f"  ✓ Stored chunk {i+1}: {heading}")
                    else:
                        logger.warning(f"  ✗ Failed to store chunk {i+1}")

            total_chunks += len(chunks)

        except Exception as e:
            logger.error(f"  ✗ Error processing {md_file.name}: {e}")
            continue

    # Summary
    logger.success(f"\n{'DRY RUN: Would ingest' if dry_run else 'Successfully ingested'} {total_chunks} chunks from {len(md_files)} file(s)")

    if not dry_run:
        stats = memory.get_memory_stats()
        logger.info(f"Memory database stats: {stats.get('total_memories')} total memories")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Bulk ingest markdown files into GLaDOS memory system"
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing markdown files to ingest'
    )
    parser.add_argument(
        '--memory-path',
        default='data/memory',
        help='Path to memory database (default: data/memory)'
    )
    parser.add_argument(
        '--config',
        help='Path to GLaDOS config YAML to load memory settings'
    )
    parser.add_argument(
        '--max-chunk-size',
        type=int,
        default=400,
        help='Maximum words per chunk (default: 400)'
    )
    parser.add_argument(
        '--memory-type',
        default='semantic',
        choices=['semantic', 'episodic', 'procedural'],
        help='Type of memory to store (default: semantic)'
    )
    parser.add_argument(
        '--speaker',
        default='system',
        help='Speaker label for memories (default: system)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be ingested without actually storing'
    )

    args = parser.parse_args()

    success = ingest_markdown_files(
        input_dir=args.input_dir,
        memory_path=args.memory_path,
        config_path=args.config,
        max_chunk_size=args.max_chunk_size,
        memory_type=args.memory_type,
        speaker=args.speaker,
        dry_run=args.dry_run
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
