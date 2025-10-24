"""
Memory Core module for GLaDOS voice assistant.

This module provides persistent memory functionality using vector embeddings
and semantic search to enhance conversation context and user experience.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import lancedb
import numpy as np
from loguru import logger

from .embedding_providers import EmbeddingProvider, create_embedding_provider, get_recommended_provider_for_device


class MemoryEntry(NamedTuple):
    """Represents a single memory entry with metadata."""
    id: str
    text: str
    embedding: list[float]
    timestamp: float
    memory_type: str
    speaker: str
    metadata: dict[str, Any] | None = None


class MemoryCore:
    """
    Manages persistent memory for GLaDOS using vector embeddings and semantic search.

    This class handles:
    - Embedding generation using lightweight sentence transformers
    - Vector storage with LanceDB for efficient similarity search
    - Memory retrieval based on semantic similarity
    - Different memory types (episodic, semantic, procedural)
    """

    def __init__(
        self,
        memory_path: str | Path = "data/memory",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "sentence_transformers",
        max_retrievals: int = 5,
        similarity_threshold: float = 0.7,
        enable_memory: bool = True,
        auto_select_provider: bool = False,
        **embedding_kwargs: Any,
    ) -> None:
        """
        Initialize the Memory Core with embedding model and vector database.

        Args:
            memory_path: Path to store the memory database
            embedding_model: Model name/identifier for embeddings
            embedding_provider: Provider type ("sentence_transformers", "embeddinggemma")
            max_retrievals: Maximum number of memories to retrieve
            similarity_threshold: Minimum similarity score for retrieval
            enable_memory: Whether memory functionality is enabled
            auto_select_provider: Automatically select best provider for hardware
            **embedding_kwargs: Additional kwargs for embedding provider
        """
        self.memory_path = Path(memory_path)
        self.max_retrievals = max_retrievals
        self.similarity_threshold = similarity_threshold
        self.enable_memory = enable_memory

        if not self.enable_memory:
            logger.info("Memory Core: Memory functionality disabled")
            return

        # Ensure memory directory exists
        self.memory_path.mkdir(parents=True, exist_ok=True)

        # Auto-select provider if requested
        if auto_select_provider:
            embedding_provider, embedding_model = get_recommended_provider_for_device()
            logger.info(f"Memory Core: Auto-selected {embedding_provider} with {embedding_model}")

        # Initialize embedding provider
        logger.info(f"Memory Core: Loading {embedding_provider} with model {embedding_model}")
        try:
            self.encoder: EmbeddingProvider = create_embedding_provider(
                provider_type=embedding_provider,
                model_name=embedding_model,
                **embedding_kwargs
            )
            self.embedding_dim = self.encoder.get_embedding_dimension()
            logger.success(f"Memory Core: Loaded embedding provider (dim={self.embedding_dim})")
        except Exception as e:
            logger.error(f"Memory Core: Failed to load embedding provider: {e}")
            self.enable_memory = False
            return

        # Initialize vector database
        try:
            self.db = lancedb.connect(str(self.memory_path))
            self._initialize_table()
            logger.success("Memory Core: Vector database initialized")
        except Exception as e:
            logger.error(f"Memory Core: Failed to initialize vector database: {e}")
            self.enable_memory = False

    def _initialize_table(self) -> None:
        """Initialize or connect to the memory table."""
        try:
            # Try to open existing table
            self.table = self.db.open_table("memories")
            logger.debug("Memory Core: Connected to existing memory table")
        except Exception:
            # Create new table with dummy data to establish schema
            logger.info("Memory Core: Creating new memory table")
            dummy_data = [{
                "id": "init",
                "text": "initialization",
                "embedding": [0.0] * self.embedding_dim,
                "timestamp": time.time(),
                "memory_type": "system",
                "speaker": "system",
                "metadata": "{}"
            }]
            # Create table with data (schema inferred from data structure)
            self.table = self.db.create_table("memories", dummy_data)
            # Remove the dummy entry
            self.table.delete("id = 'init'")

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if not self.enable_memory:
            return []

        try:
            embedding = self.encoder.encode(text)
            # Handle single text case
            if embedding.ndim == 1:
                return embedding.tolist()
            else:
                return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Memory Core: Failed to generate embedding: {e}")
            return []

    def store_memory(
        self,
        text: str,
        memory_type: str = "episodic",
        speaker: str = "user",
        metadata: dict[str, Any] | None = None,
        deduplicate: bool = True,
        similarity_threshold: float = 0.95
    ) -> str:
        """
        Store a new memory entry.

        Args:
            text: The text content to store
            memory_type: Type of memory (episodic, semantic, procedural)
            speaker: Who said/did this (user, assistant, system)
            metadata: Additional metadata dictionary
            deduplicate: Skip storage if very similar memory exists
            similarity_threshold: Threshold for deduplication (0.95 = 95% similar)

        Returns:
            The memory ID if successful, empty string if failed
        """
        if not self.enable_memory or not text.strip():
            return ""

        try:
            # Check for duplicates if enabled
            if deduplicate:
                existing = self.retrieve_memories(text, max_results=1)
                if existing and existing[0]["similarity"] >= similarity_threshold:
                    logger.debug(f"Memory Core: Skipping duplicate (similarity={existing[0]['similarity']:.2f}): {text[:50]}...")
                    return existing[0]["id"]  # Return existing ID

            # Generate embedding
            embedding = self.embed_text(text)
            if not embedding:
                return ""

            # Create memory entry
            memory_id = f"{memory_type}_{speaker}_{int(time.time() * 1000)}"
            entry = {
                "id": memory_id,
                "text": text.strip(),
                "embedding": embedding,
                "timestamp": time.time(),
                "memory_type": memory_type,
                "speaker": speaker,
                "metadata": json.dumps(metadata or {})
            }

            # Store in database
            self.table.add([entry])
            logger.debug(f"Memory Core: Stored memory {memory_id}: {text[:50]}...")
            return memory_id

        except Exception as e:
            logger.error(f"Memory Core: Failed to store memory: {e}")
            return ""

    def retrieve_memories(
        self,
        query: str,
        memory_types: list[str] | None = None,
        speaker_filter: str | None = None,
        max_results: int | None = None,
        exclude_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant memories based on semantic similarity.

        Args:
            query: Query text to search for
            memory_types: Filter by memory types (None for all)
            speaker_filter: Filter by speaker (None for all)
            max_results: Override default max retrievals
            exclude_ids: List of memory IDs to exclude from results

        Returns:
            List of memory dictionaries with similarity scores
        """
        if not self.enable_memory or not query.strip():
            return []

        try:
            # Generate query embedding
            query_embedding = self.embed_text(query)
            if not query_embedding:
                return []

            # Build filter condition
            filter_parts = []
            if memory_types:
                type_filter = " OR ".join([f"memory_type = '{t}'" for t in memory_types])
                filter_parts.append(f"({type_filter})")
            if speaker_filter:
                filter_parts.append(f"speaker = '{speaker_filter}'")
            if exclude_ids:
                # Exclude specific memory IDs
                exclude_filter = " AND ".join([f"id != '{eid}'" for eid in exclude_ids])
                filter_parts.append(f"({exclude_filter})")

            filter_condition = " AND ".join(filter_parts) if filter_parts else None

            # Perform vector search with cosine metric
            max_results = max_results or self.max_retrievals
            search = self.table.search(query_embedding).metric("cosine").limit(max_results)

            if filter_condition:
                search = search.where(filter_condition)

            results = search.to_list()

            # Filter by similarity threshold and format results
            relevant_memories = []
            for result in results:
                # With cosine metric, _distance is cosine distance (1 - cosine_similarity)
                # So similarity = 1 - distance
                similarity = 1.0 - result["_distance"]

                if similarity >= self.similarity_threshold:
                    memory = {
                        "id": result["id"],
                        "text": result["text"],
                        "similarity": similarity,
                        "timestamp": result["timestamp"],
                        "memory_type": result["memory_type"],
                        "speaker": result["speaker"],
                        "metadata": json.loads(result["metadata"]) if result["metadata"] else {}
                    }
                    relevant_memories.append(memory)

            # Sort by hybrid score: similarity + recency boost
            # Only boost episodic memories (semantic/procedural are timeless facts)
            import time
            current_time = time.time()
            for mem in relevant_memories:
                if mem["memory_type"] == "episodic":
                    age_hours = (current_time - mem["timestamp"]) / 3600
                    recency_boost = max(0, 0.1 * (1 - min(age_hours / 168, 1)))  # Decay over 1 week
                else:
                    recency_boost = 0  # No boost for semantic/procedural facts
                mem["hybrid_score"] = mem["similarity"] + recency_boost

            relevant_memories.sort(key=lambda x: x["hybrid_score"], reverse=True)

            if relevant_memories:
                logger.debug(f"Memory Core: Retrieved {len(relevant_memories)} memories for query: {query[:50]}...")

            return relevant_memories

        except Exception as e:
            logger.error(f"Memory Core: Failed to retrieve memories: {e}")
            return []

    def get_active_tasks_and_reminders(self, max_results: int = 10) -> list[dict[str, Any]]:
        """
        Retrieve active tasks and reminders regardless of query relevance.
        These should be proactively surfaced to keep the user informed.

        Returns:
            List of task/reminder memories with status='active'
        """
        if not self.enable_memory:
            return []

        try:
            import json

            # Get all recent episodic memories
            all_memories = self.table.search().limit(100).to_list()

            active_items = []
            for mem in all_memories:
                # Parse metadata (it's stored as JSON string)
                metadata = mem.get('metadata', '{}')
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}

                # Filter for active tasks/reminders
                semantic_type = metadata.get('semantic_type', '')
                status = metadata.get('status', '')

                if semantic_type in ['task', 'reminder'] and status == 'active':
                    mem['parsed_metadata'] = metadata
                    active_items.append(mem)

            # Return most recent active items
            return active_items[:max_results]

        except Exception as e:
            logger.error(f"Memory Core: Failed to retrieve active tasks/reminders: {e}")
            return []

    def get_recent_memories(
        self,
        hours: int = 24,
        memory_types: list[str] | None = None,
        max_results: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get recent memories within specified time window.

        Args:
            hours: Number of hours to look back
            memory_types: Filter by memory types
            max_results: Maximum number of results

        Returns:
            List of recent memory dictionaries
        """
        if not self.enable_memory:
            return []

        try:
            cutoff_time = time.time() - (hours * 3600)

            # Build filter
            filter_parts = [f"timestamp >= {cutoff_time}"]
            if memory_types:
                type_filter = " OR ".join([f"memory_type = '{t}'" for t in memory_types])
                filter_parts.append(f"({type_filter})")

            filter_condition = " AND ".join(filter_parts)

            # Query recent memories
            max_results = max_results or self.max_retrievals
            results = self.table.search().where(filter_condition).limit(max_results).to_list()

            # Format and sort by timestamp (newest first)
            memories = []
            for result in results:
                memory = {
                    "id": result["id"],
                    "text": result["text"],
                    "timestamp": result["timestamp"],
                    "memory_type": result["memory_type"],
                    "speaker": result["speaker"],
                    "metadata": json.loads(result["metadata"]) if result["metadata"] else {}
                }
                memories.append(memory)

            memories.sort(key=lambda x: x["timestamp"], reverse=True)
            return memories

        except Exception as e:
            logger.error(f"Memory Core: Failed to get recent memories: {e}")
            return []

    def clear_memory(self, memory_type: str | None = None) -> bool:
        """
        Clear memories, optionally filtered by type.

        Args:
            memory_type: Specific memory type to clear (None for all)

        Returns:
            True if successful
        """
        if not self.enable_memory:
            return True

        try:
            if memory_type:
                self.table.delete(f"memory_type = '{memory_type}'")
                logger.info(f"Memory Core: Cleared {memory_type} memories")
            else:
                # Clear entire table
                self.table.delete("timestamp > 0")
                logger.info("Memory Core: Cleared all memories")

            return True

        except Exception as e:
            logger.error(f"Memory Core: Failed to clear memory: {e}")
            return False

    def get_memory_stats(self) -> dict[str, Any]:
        """Get statistics about stored memories."""
        if not self.enable_memory:
            return {"enabled": False}

        try:
            # Count total memories
            total_count = self.table.count_rows()

            # Count by type
            type_counts = {}
            speaker_counts = {}

            if total_count > 0:
                all_memories = self.table.search().limit(total_count).to_list()

                for memory in all_memories:
                    mem_type = memory["memory_type"]
                    speaker = memory["speaker"]

                    type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
                    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

            return {
                "enabled": True,
                "total_memories": total_count,
                "by_type": type_counts,
                "by_speaker": speaker_counts,
                "embedding_model": self.encoder.get_model_name(),
                "embedding_dimension": self.embedding_dim
            }

        except Exception as e:
            logger.error(f"Memory Core: Failed to get stats: {e}")
            return {"enabled": True, "error": str(e)}

    def format_context_for_llm(self, memories: list[dict[str, Any]]) -> str:
        """
        Format retrieved memories as context for LLM prompt.

        Args:
            memories: List of memory dictionaries

        Returns:
            Formatted context string
        """
        if not memories:
            return ""

        context_parts = ["Previous relevant context:"]

        # Parse metadata to check for semantic types
        import json

        for memory in memories[:self.max_retrievals]:  # Limit context size
            speaker = memory["speaker"]
            text = memory["text"]
            memory_type = memory["memory_type"]

            # Parse metadata if present
            metadata = memory.get('metadata', '{}')
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}

            semantic_type = metadata.get('semantic_type', '')

            # Format based on semantic type first, then memory type and speaker
            if semantic_type == 'task':
                context_parts.append(f"ACTIVE TASK: {text}")
            elif semantic_type == 'reminder':
                context_parts.append(f"ACTIVE REMINDER: {text}")
            elif semantic_type == 'preference':
                context_parts.append(f"User preference: {text}")
            elif speaker == "user":
                context_parts.append(f"User previously said: \"{text}\"")
            elif speaker == "assistant":
                context_parts.append(f"You previously responded: \"{text}\"")
            elif memory_type == "semantic":
                context_parts.append(f"Important fact: {text}")
            else:
                context_parts.append(f"{speaker.title()}: \"{text}\"")

        return "\n".join(context_parts)

    def find_duplicate_clusters(
        self,
        similarity_threshold: float = 0.90,
        batch_size: int = 100
    ) -> list[list[dict[str, Any]]]:
        """
        Find clusters of duplicate or near-duplicate memories.

        Args:
            similarity_threshold: Threshold for considering memories duplicates
            batch_size: Number of memories to process at once

        Returns:
            List of clusters, where each cluster is a list of duplicate memories
        """
        if not self.enable_memory:
            return []

        try:
            # Get all memories
            total_count = self.table.count_rows()
            if total_count == 0:
                return []

            logger.info(f"Memory Core: Scanning {total_count} memories for duplicates...")

            all_memories = self.table.search().limit(total_count).to_list()

            # Build clusters using similarity search
            clusters = []
            processed_ids = set()

            for i, memory in enumerate(all_memories):
                if memory["id"] in processed_ids:
                    continue

                # Search for similar memories
                similar = self.retrieve_memories(
                    query=memory["text"],
                    max_results=50,  # Check up to 50 similar items
                    exclude_ids=list(processed_ids)
                )

                # Filter by threshold and exclude self
                duplicates = [
                    mem for mem in similar
                    if mem["id"] != memory["id"] and mem["similarity"] >= similarity_threshold
                ]

                if duplicates:
                    # Create cluster with original + duplicates
                    cluster = [memory] + duplicates
                    clusters.append(cluster)

                    # Mark all as processed
                    processed_ids.add(memory["id"])
                    for dup in duplicates:
                        processed_ids.add(dup["id"])

                # Progress logging
                if (i + 1) % batch_size == 0:
                    logger.debug(f"Memory Core: Processed {i + 1}/{total_count} memories...")

            logger.info(f"Memory Core: Found {len(clusters)} duplicate clusters")
            return clusters

        except Exception as e:
            logger.error(f"Memory Core: Failed to find duplicates: {e}")
            return []

    def deduplicate_memories(
        self,
        similarity_threshold: float = 0.90,
        strategy: str = "keep_newest",
        dry_run: bool = False
    ) -> dict[str, Any]:
        """
        Remove duplicate memories from the database.

        Args:
            similarity_threshold: Threshold for considering memories duplicates (0.90 = 90% similar)
            strategy: How to choose which duplicate to keep:
                - "keep_newest": Keep the most recent memory
                - "keep_oldest": Keep the earliest memory
                - "keep_longest": Keep the memory with most text
            dry_run: If True, only report what would be deleted without deleting

        Returns:
            Dictionary with deduplication statistics
        """
        if not self.enable_memory:
            return {"enabled": False}

        try:
            # Find duplicate clusters
            clusters = self.find_duplicate_clusters(similarity_threshold=similarity_threshold)

            if not clusters:
                logger.info("Memory Core: No duplicates found")
                return {
                    "clusters_found": 0,
                    "memories_removed": 0,
                    "memories_kept": 0,
                    "dry_run": dry_run
                }

            memories_to_remove = []
            memories_kept = []

            for cluster in clusters:
                # Choose which memory to keep based on strategy
                if strategy == "keep_newest":
                    keeper = max(cluster, key=lambda m: m.get("timestamp", 0))
                elif strategy == "keep_oldest":
                    keeper = min(cluster, key=lambda m: m.get("timestamp", float("inf")))
                elif strategy == "keep_longest":
                    keeper = max(cluster, key=lambda m: len(m.get("text", "")))
                else:
                    logger.warning(f"Unknown strategy '{strategy}', using 'keep_newest'")
                    keeper = max(cluster, key=lambda m: m.get("timestamp", 0))

                memories_kept.append(keeper)

                # Mark others for removal
                for mem in cluster:
                    if mem["id"] != keeper["id"]:
                        memories_to_remove.append(mem)

            # Perform deletion if not dry run
            removed_count = 0
            if not dry_run and memories_to_remove:
                for mem in memories_to_remove:
                    try:
                        self.table.delete(f"id = '{mem['id']}'")
                        removed_count += 1
                    except Exception as e:
                        logger.error(f"Memory Core: Failed to delete {mem['id']}: {e}")

                logger.success(f"Memory Core: Removed {removed_count} duplicate memories")
            else:
                removed_count = len(memories_to_remove)
                logger.info(f"Memory Core: [DRY RUN] Would remove {removed_count} duplicate memories")

            # Prepare detailed report
            stats = {
                "clusters_found": len(clusters),
                "memories_removed": removed_count,
                "memories_kept": len(memories_kept),
                "dry_run": dry_run,
                "strategy": strategy,
                "threshold": similarity_threshold
            }

            # Add sample duplicates for inspection
            if dry_run and clusters:
                stats["sample_clusters"] = []
                for cluster in clusters[:3]:  # Show first 3 clusters
                    stats["sample_clusters"].append([
                        {
                            "id": m["id"],
                            "text": m["text"][:100] + "..." if len(m["text"]) > 100 else m["text"],
                            "timestamp": datetime.fromtimestamp(m.get("timestamp", 0)).isoformat(),
                            "similarity": m.get("similarity", 1.0)
                        }
                        for m in cluster
                    ])

            return stats

        except Exception as e:
            logger.error(f"Memory Core: Failed to deduplicate: {e}")
            return {"error": str(e)}

    def consolidate_old_memories(
        self,
        days_threshold: int = 30,
        similarity_threshold: float = 0.85,
        dry_run: bool = False
    ) -> dict[str, Any]:
        """
        Consolidate old episodic memories that are similar into single entries.

        This helps reduce memory clutter for frequently discussed topics.

        Args:
            days_threshold: Only consolidate memories older than this many days
            similarity_threshold: Threshold for grouping similar memories
            dry_run: If True, only report what would be consolidated

        Returns:
            Dictionary with consolidation statistics
        """
        if not self.enable_memory:
            return {"enabled": False}

        try:
            cutoff_time = time.time() - (days_threshold * 86400)

            # Get old episodic memories
            old_memories = []
            all_memories = self.table.search().where(
                f"timestamp < {cutoff_time} AND memory_type = 'episodic'"
            ).limit(10000).to_list()

            for mem in all_memories:
                old_memories.append({
                    "id": mem["id"],
                    "text": mem["text"],
                    "timestamp": mem["timestamp"],
                    "memory_type": mem["memory_type"],
                    "speaker": mem["speaker"],
                    "metadata": json.loads(mem["metadata"]) if mem["metadata"] else {}
                })

            if not old_memories:
                logger.info(f"Memory Core: No episodic memories older than {days_threshold} days")
                return {"consolidated": 0, "removed": 0, "dry_run": dry_run}

            logger.info(f"Memory Core: Found {len(old_memories)} old episodic memories to consider")

            # Find similar clusters among old memories
            consolidated = 0
            removed = 0

            # This would need more sophisticated implementation for production
            # For now, just use the existing deduplication on old memories
            logger.info("Memory Core: Consolidation uses same logic as deduplication")
            return {"note": "Use deduplicate_memories() for old memory cleanup"}

        except Exception as e:
            logger.error(f"Memory Core: Failed to consolidate: {e}")
            return {"error": str(e)}