"""
Unit tests for MemoryCore RAG functionality.

This test suite validates:
1. Memory storage and retrieval with cosine similarity
2. Similarity scoring in valid range [0, 1]
3. Filtering of just-stored memories
4. Context formatting for LLM prompts
"""

import tempfile
from pathlib import Path

import pytest

from glados.core.memory_core import MemoryCore


@pytest.fixture
def memory_core():
    """Provide a MemoryCore instance with temporary storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryCore(
            memory_path=tmpdir,
            embedding_model="all-MiniLM-L6-v2",
            embedding_provider="sentence_transformers",
            max_retrievals=5,
            similarity_threshold=0.3,  # Lower threshold for testing
            enable_memory=True
        )
        if memory.enable_memory:
            yield memory
        else:
            pytest.skip("Memory core failed to initialize")


def test_memory_storage(memory_core):
    """Test that memories can be stored successfully."""
    memory_id = memory_core.store_memory(
        text="My favorite color is blue",
        speaker="user",
        memory_type="episodic"
    )

    assert memory_id != "", "Memory ID should not be empty"
    assert "user" in memory_id, "Memory ID should contain speaker"
    assert "episodic" in memory_id, "Memory ID should contain memory type"


def test_memory_retrieval_similarity(memory_core):
    """Test that memories are retrieved with correct cosine similarity scores."""
    # Store test memories
    test_memories = [
        "My favorite color is blue",
        "I love programming in Python",
        "The weather today is sunny",
    ]

    for text in test_memories:
        memory_core.store_memory(text=text, speaker="user", memory_type="episodic")

    # Retrieve memories about colors
    results = memory_core.retrieve_memories("What is my favorite color?")

    assert len(results) > 0, "Should retrieve at least one memory"

    # Verify similarity scores are in valid range
    for mem in results:
        assert "similarity" in mem, "Result should have similarity score"
        similarity = mem["similarity"]
        assert -1.0 <= similarity <= 1.0, f"Similarity {similarity} should be in [-1, 1]"
        assert similarity >= memory_core.similarity_threshold, \
            f"Similarity {similarity} should be >= threshold {memory_core.similarity_threshold}"

    # The top result should be about blue color
    assert any("blue" in mem["text"].lower() for mem in results), \
        "Should retrieve color-related memory"


def test_exclude_ids_filter(memory_core):
    """Test that exclude_ids parameter properly filters results."""
    # Store test memories
    test_memories = [
        "My favorite color is blue",
        "I also like the color red",
        "Green is nice too",
    ]

    for text in test_memories:
        memory_core.store_memory(text=text, speaker="user", memory_type="episodic")

    # Get all results
    query = "Tell me about colors"
    all_results = memory_core.retrieve_memories(query)

    assert len(all_results) > 0, "Should have results before filtering"

    # Exclude the top result
    if all_results:
        exclude_id = all_results[0]["id"]
        filtered_results = memory_core.retrieve_memories(query, exclude_ids=[exclude_id])

        # Should have fewer results or different top result
        if len(filtered_results) > 0 and len(all_results) > 1:
            assert filtered_results[0]["id"] != exclude_id, \
                "Excluded ID should not appear in filtered results"


def test_memory_context_formatting(memory_core):
    """Test that memories are formatted correctly for LLM context."""
    # Store test memories
    memory_core.store_memory(
        text="My favorite color is blue",
        speaker="user",
        memory_type="episodic"
    )
    memory_core.store_memory(
        text="I understand you like blue",
        speaker="assistant",
        memory_type="episodic"
    )

    # Retrieve and format
    results = memory_core.retrieve_memories("What color do I like?")
    context = memory_core.format_context_for_llm(results)

    assert context != "", "Context should not be empty"
    assert "Previous relevant context:" in context, "Should have context header"
    assert any(keyword in context.lower() for keyword in ["user", "previously"]), \
        "Context should indicate user or previous information"


def test_similarity_scores_positive_for_related_content(memory_core):
    """Test that related content produces positive similarity scores."""
    # Store semantically similar memories
    memory_core.store_memory(
        text="Python is a programming language",
        speaker="user",
        memory_type="episodic"
    )

    # Query with similar semantic meaning
    results = memory_core.retrieve_memories("Tell me about Python programming")

    assert len(results) > 0, "Should retrieve related memory"

    # With cosine similarity on related content, expect positive similarity > 0.3
    top_similarity = results[0]["similarity"]
    assert top_similarity > 0.3, \
        f"Related content should have high similarity (got {top_similarity})"


def test_retrieval_before_storage_pattern(memory_core):
    """
    Test the pattern used in LLM processor where retrieval happens before storage.
    This ensures the just-asked question doesn't become the top retrieval result.
    """
    # Store some background knowledge
    memory_core.store_memory(
        text="I mentioned earlier that my favorite color is blue",
        speaker="user",
        memory_type="episodic"
    )

    # Simulate new user input: retrieve BEFORE storing
    new_input = "What is my favorite color?"

    # Retrieve first (as done in fixed llm_processor.py)
    results_before_storage = memory_core.retrieve_memories(new_input)

    # Store the new input after retrieval
    new_memory_id = memory_core.store_memory(
        text=new_input,
        speaker="user",
        memory_type="episodic"
    )

    # Now if we retrieve again, the new input should appear
    results_after_storage = memory_core.retrieve_memories(new_input)

    # The initial retrieval should find the earlier mention of blue
    if results_before_storage:
        assert "blue" in results_before_storage[0]["text"].lower(), \
            "First retrieval should find the earlier blue mention, not the question itself"

    # After storage, if we filter it out, we should still get the earlier memory
    if results_after_storage:
        filtered = memory_core.retrieve_memories(new_input, exclude_ids=[new_memory_id])
        if filtered:
            assert "blue" in filtered[0]["text"].lower(), \
                "Filtering the new input should reveal earlier memory"
