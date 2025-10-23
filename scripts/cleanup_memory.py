#!/usr/bin/env python3
"""
Memory cleanup and deduplication script for GLaDOS.

This script helps maintain a streamlined memory database by:
- Finding and removing duplicate memories
- Showing statistics about memory usage
- Running dry-run mode to preview changes before applying

Usage:
    # Dry run - see what would be removed
    python scripts/cleanup_memory.py --dry-run

    # Remove duplicates (90% similarity threshold)
    python scripts/cleanup_memory.py

    # Aggressive deduplication (85% similarity)
    python scripts/cleanup_memory.py --threshold 0.85

    # Keep oldest memories instead of newest
    python scripts/cleanup_memory.py --strategy keep_oldest

    # Use custom config file
    python scripts/cleanup_memory.py --config configs/glados_config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glados.core.memory_core import MemoryCore
from loguru import logger


def load_memory_config(config_path: str | None) -> dict:
    """Load memory configuration from YAML config file."""
    if not config_path:
        # Use defaults
        return {
            "memory_path": "data/memory",
            "embedding_provider": "sentence_transformers",
            "embedding_model": "all-MiniLM-L6-v2",
            "enable_memory": True,
        }

    import yaml
    from glados.core.engine import GladosConfig

    config = GladosConfig.from_yaml(config_path)

    return {
        "memory_path": config.memory_path,
        "embedding_provider": config.memory_embedding_provider,
        "embedding_model": config.memory_embedding_model,
        "enable_memory": config.memory_enabled,
        # Pass through provider-specific settings
        "prompt_name": config.memory_gemma_prompt_name,
        "truncate_dim": config.memory_gemma_truncate_dim,
        "ollama_url": config.memory_ollama_url,
        "auto_pull": config.memory_ollama_auto_pull,
        "max_retries": config.memory_ollama_max_retries,
    }


def print_stats(stats: dict) -> None:
    """Pretty print memory statistics."""
    print("\n" + "=" * 60)
    print("MEMORY DATABASE STATISTICS")
    print("=" * 60)

    if not stats.get("enabled"):
        print("❌ Memory is disabled")
        return

    print(f"📊 Total Memories: {stats.get('total_memories', 0)}")
    print(f"🔧 Embedding Model: {stats.get('embedding_model', 'unknown')}")
    print(f"📐 Embedding Dimension: {stats.get('embedding_dimension', 'unknown')}")

    by_type = stats.get("by_type", {})
    if by_type:
        print("\n📁 By Type:")
        for mem_type, count in sorted(by_type.items()):
            print(f"   - {mem_type}: {count}")

    by_speaker = stats.get("by_speaker", {})
    if by_speaker:
        print("\n🗣️  By Speaker:")
        for speaker, count in sorted(by_speaker.items()):
            print(f"   - {speaker}: {count}")

    print("=" * 60 + "\n")


def print_dedup_results(results: dict) -> None:
    """Pretty print deduplication results."""
    print("\n" + "=" * 60)
    print("DEDUPLICATION RESULTS")
    print("=" * 60)

    if results.get("error"):
        print(f"❌ Error: {results['error']}")
        return

    dry_run = results.get("dry_run", False)
    prefix = "[DRY RUN] " if dry_run else ""

    print(f"🔍 Clusters Found: {results.get('clusters_found', 0)}")
    print(f"✅ Memories Kept: {results.get('memories_kept', 0)}")

    if dry_run:
        print(f"🗑️  Would Remove: {results.get('memories_removed', 0)}")
    else:
        print(f"🗑️  Removed: {results.get('memories_removed', 0)}")

    print(f"⚙️  Strategy: {results.get('strategy', 'unknown')}")
    print(f"📊 Threshold: {results.get('threshold', 0.0):.2f}")

    # Show sample clusters
    sample_clusters = results.get("sample_clusters", [])
    if sample_clusters and dry_run:
        print("\n📋 Sample Duplicate Clusters (first 3):")
        for i, cluster in enumerate(sample_clusters, 1):
            print(f"\n   Cluster {i} ({len(cluster)} duplicates):")
            for j, mem in enumerate(cluster):
                timestamp = mem.get("timestamp", "unknown")
                similarity = mem.get("similarity", 1.0)
                text = mem.get("text", "")
                marker = "→ KEEP" if j == 0 else "  DELETE"
                print(f"      {marker} [{similarity:.2f}] {timestamp}")
                print(f"           {text}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up and deduplicate GLaDOS memory database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to GLaDOS config YAML file (uses defaults if not specified)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Similarity threshold for considering memories duplicates (0.0-1.0, default: 0.90)"
    )

    parser.add_argument(
        "--strategy",
        choices=["keep_newest", "keep_oldest", "keep_longest"],
        default="keep_newest",
        help="Strategy for choosing which duplicate to keep (default: keep_newest)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing anything"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show memory statistics, don't perform deduplication"
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (automatic yes)"
    )

    args = parser.parse_args()

    # Load configuration
    logger.info("Loading memory configuration...")
    mem_config = load_memory_config(args.config)

    # Initialize memory core
    logger.info(f"Initializing memory from {mem_config['memory_path']}...")

    # Prepare provider-specific kwargs
    embedding_kwargs = {}
    if mem_config.get("prompt_name"):
        embedding_kwargs["prompt_name"] = mem_config["prompt_name"]
    if mem_config.get("truncate_dim"):
        embedding_kwargs["truncate_dim"] = mem_config["truncate_dim"]
    if mem_config.get("ollama_url"):
        embedding_kwargs["ollama_url"] = mem_config["ollama_url"]
    if mem_config.get("auto_pull") is not None:
        embedding_kwargs["auto_pull"] = mem_config["auto_pull"]
    if mem_config.get("max_retries"):
        embedding_kwargs["max_retries"] = mem_config["max_retries"]

    memory = MemoryCore(
        memory_path=mem_config["memory_path"],
        embedding_model=mem_config["embedding_model"],
        embedding_provider=mem_config["embedding_provider"],
        enable_memory=mem_config["enable_memory"],
        **embedding_kwargs
    )

    if not memory.enable_memory:
        logger.error("Memory is disabled, cannot proceed")
        return 1

    # Show statistics
    stats = memory.get_memory_stats()
    print_stats(stats)

    if args.stats_only:
        logger.info("Statistics only mode - exiting")
        return 0

    total_memories = stats.get("total_memories", 0)
    if total_memories == 0:
        logger.info("No memories to deduplicate")
        return 0

    # Confirmation prompt (unless dry-run or --yes)
    if not args.dry_run and not args.yes:
        print(f"⚠️  About to deduplicate {total_memories} memories")
        print(f"   Threshold: {args.threshold:.2f}")
        print(f"   Strategy: {args.strategy}")
        response = input("\nProceed? (yes/no): ").strip().lower()
        if response not in ("yes", "y"):
            logger.info("Cancelled by user")
            return 0

    # Run deduplication
    logger.info("Starting deduplication scan...")
    results = memory.deduplicate_memories(
        similarity_threshold=args.threshold,
        strategy=args.strategy,
        dry_run=args.dry_run
    )

    # Show results
    print_dedup_results(results)

    # Show updated stats if not dry run
    if not args.dry_run and results.get("memories_removed", 0) > 0:
        logger.info("Fetching updated statistics...")
        updated_stats = memory.get_memory_stats()
        print("\nUPDATED STATISTICS:")
        print_stats(updated_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
