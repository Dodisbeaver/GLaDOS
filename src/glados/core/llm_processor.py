# --- llm_processor.py ---
import json
import queue
import re
import threading
import time
from typing import Any, ClassVar

from loguru import logger
from pydantic import HttpUrl  # If HttpUrl is used by config
import requests

from .memory_core import MemoryCore


class LanguageModelProcessor:
    """
    A thread that processes text input for a language model, streaming responses and sending them to TTS.
    This class is designed to run in a separate thread, continuously checking for new text to process
    until a shutdown event is set. It handles conversation history, manages streaming responses,
    and sends synthesized sentences to a TTS queue.
    """

    PUNCTUATION_SET: ClassVar[set[str]] = {".", "!", "?", ":", ";", "?!", "\n", "\n\n"}

    def __init__(
        self,
        llm_input_queue: queue.Queue[str],
        tts_input_queue: queue.Queue[str],
        conversation_history: list[dict[str, str]],  # Shared
        completion_url: HttpUrl,
        model_name: str,  # Renamed from 'model' to avoid conflict
        api_key: str | None,
        processing_active_event: threading.Event,  # To check if we should stop streaming
        shutdown_event: threading.Event,
        pause_time: float = 0.05,
        memory_core: MemoryCore | None = None,
        store_user_inputs: bool = True,
        store_assistant_responses: bool = True,
    ) -> None:
        self.llm_input_queue = llm_input_queue
        self.tts_input_queue = tts_input_queue
        self.conversation_history = conversation_history
        self.completion_url = completion_url
        self.model_name = model_name
        self.api_key = api_key
        self.processing_active_event = processing_active_event
        self.shutdown_event = shutdown_event
        self.pause_time = pause_time
        self.memory_core = memory_core
        self.store_user_inputs = store_user_inputs
        self.store_assistant_responses = store_assistant_responses

        self.prompt_headers = {"Content-Type": "application/json"}

        # State for filtering chain-of-thought tags
        self._inside_think_tag = False
        self._tag_buffer = ""
        if api_key:
            self.prompt_headers["Authorization"] = f"Bearer {api_key}"

    def _clean_raw_bytes(self, line: bytes) -> dict[str, str] | None:
        """
        Clean and parse a raw byte line from the LLM response.
        Handles both OpenAI and Ollama formats, returning a dictionary or None if parsing fails.

        Args:
            line (bytes): The raw byte line from the LLM response.
        Returns:
            dict[str, str] | None: Parsed JSON dictionary or None if parsing fails.
        """
        try:
            # Handle OpenAI format
            if line.startswith(b"data: "):
                json_str = line.decode("utf-8")[6:]
                if json_str.strip() == "[DONE]":  # Handle OpenAI [DONE] marker
                    return {"done_marker": "True"}
                parsed_json: dict[str, Any] = json.loads(json_str)
                return parsed_json
            # Handle Ollama format
            else:
                parsed_json = json.loads(line.decode("utf-8"))
                if isinstance(parsed_json, dict):
                    return parsed_json
                return None
        except json.JSONDecodeError:
            # If it's not JSON, it might be Ollama's final summary object which isn't part of the stream
            # Or just noise.
            logger.trace(
                f"LLM Processor: Failed to parse non-JSON server response line: "
                f"{line[:100].decode('utf-8', errors='replace')}"
            )  # Log only a part
            return None
        except Exception as e:
            logger.warning(
                f"LLM Processor: Failed to parse server response: {e} for line: "
                f"{line[:100].decode('utf-8', errors='replace')}"
            )
            return None

    def _reset_think_filter_state(self) -> None:
        """Reset the think tag filter state. Should be called on request start and error recovery."""
        self._inside_think_tag = False
        self._tag_buffer = ""

    def _filter_think_tags(self, chunk: str) -> str:
        """Filter out <think>...</think> tags from streaming content.

        Maintains state across chunks to handle tags split across multiple chunks.
        Uses efficient parsing to avoid performance issues and memory leaks.
        """
        if not chunk:
            return chunk

        result = []
        i = 0

        while i < len(chunk):
            if not self._inside_think_tag:
                # Look for opening tag
                tag_start = chunk.find("<think>", i)
                if tag_start == -1:
                    # No opening tag found, output remaining content
                    result.append(chunk[i:])
                    break
                else:
                    # Output content before tag
                    result.append(chunk[i:tag_start])
                    self._inside_think_tag = True
                    i = tag_start + 7  # len("<think>")
            else:
                # Look for closing tag
                tag_end = chunk.find("</think>", i)
                if tag_end == -1:
                    # No closing tag found in this chunk, skip remaining content
                    # It will be in a future chunk
                    break
                else:
                    # Found closing tag, skip content inside think block
                    self._inside_think_tag = False
                    i = tag_end + 8  # len("</think>")

        return "".join(result)

    def _process_chunk(self, line: dict[str, Any]) -> str | None:
        # Copy from Glados._process_chunk
        if not line or not isinstance(line, dict):
            return None
        try:
            # Handle OpenAI format
            if line.get("done_marker"):  # Handle [DONE] marker
                return None
            elif "choices" in line:  # OpenAI format
                content = line.get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    return self._filter_think_tags(str(content))
                return None
            # Handle Ollama format
            else:
                content = line.get("message", {}).get("content")
                if content:
                    return self._filter_think_tags(content)
                return None
        except Exception as e:
            logger.error(f"LLM Processor: Error processing chunk: {e}, chunk: {line}")
            return None

    def _process_sentence_for_tts(self, current_sentence_parts: list[str]) -> None:
        """
        Process the current sentence parts and send the complete sentence to the TTS queue.
        Cleans up the sentence by removing unwanted characters and formatting it for TTS.
        Args:
            current_sentence_parts (list[str]): List of sentence parts to be processed.
        """
        sentence = "".join(current_sentence_parts)
        # Remove markdown formatting and parenthetical asides
        # Note: <think> tags are already filtered out during streaming
        sentence = re.sub(r"\*.*?\*|\(.*?\)", "", sentence)
        sentence = sentence.replace("\n\n", ". ").replace("\n", ". ").replace("  ", " ").replace(":", " ")

        if sentence and sentence != ".":  # Avoid sending just a period
            logger.info(f"LLM Processor: Sending to TTS queue: '{sentence}'")
            self.tts_input_queue.put(sentence)

    def run(self) -> None:
        """
        Starts the main loop for the LanguageModelProcessor thread.

        This method continuously checks the LLM input queue for text to process.
        It processes the text, sends it to the LLM API, and streams the response.
        It handles conversation history, manages streaming responses, and sends synthesized sentences
        to a TTS queue. The thread will run until the shutdown event is set, at which point it will exit gracefully.
        """
        logger.info("LanguageModelProcessor thread started.")
        while not self.shutdown_event.is_set():
            try:
                detected_text = self.llm_input_queue.get(timeout=self.pause_time)
                if not self.processing_active_event.is_set():  # Check if we were interrupted before starting
                    logger.info("LLM Processor: Interruption signal active, discarding LLM request.")
                    # Ensure EOS is sent if a previous stream was cut short by this interruption
                    # This logic might need refinement based on state. For now, assume no prior stream.
                    continue

                logger.info(f"LLM Processor: Received text for LLM: '{detected_text}'")

                # Handle special commands
                if detected_text == "__CLEAR_MEMORY__":
                    # Clear conversation history except for the system prompt
                    original_length = len(self.conversation_history)
                    # Keep only system messages (usually the first message)
                    self.conversation_history = [msg for msg in self.conversation_history if msg.get("role") == "system"]
                    cleared_count = original_length - len(self.conversation_history)
                    logger.info(f"Conversation history cleared. Removed {cleared_count} messages.")

                    # Send confirmation message to TTS
                    self.tts_input_queue.put("Memory cleared... Let's start fresh.")
                    self.tts_input_queue.put("<EOS>")
                    continue

                self.conversation_history.append({"role": "user", "content": detected_text})

                # Retrieve relevant memories for context BEFORE storing the new input
                # This prevents the just-stored message from being the top hit
                memory_context = ""
                if self.memory_core:
                    logger.info(f"LLM Processor: Retrieving memories for query: '{detected_text[:100]}...'")
                    relevant_memories = self.memory_core.retrieve_memories(
                        query=detected_text,
                        memory_types=["episodic", "semantic", "procedural"]
                    )
                    if relevant_memories:
                        memory_context = self.memory_core.format_context_for_llm(relevant_memories)
                        # Count memory types
                        type_counts = {}
                        for mem in relevant_memories:
                            mem_type = mem['memory_type']
                            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
                        type_summary = ", ".join([f"{count} {mtype}" for mtype, count in type_counts.items()])
                        logger.success(f"LLM Processor: Retrieved {len(relevant_memories)} memories ({type_summary})")
                        for i, mem in enumerate(relevant_memories[:3], 1):
                            logger.info(f"  Memory {i}: [{mem['memory_type']}] similarity={mem['similarity']:.3f} - {mem['text'][:100]}...")
                    else:
                        logger.warning(f"LLM Processor: No relevant memories found (threshold={self.memory_core.similarity_threshold})")

                # Store user input in memory if enabled (after retrieval)
                if self.memory_core and self.store_user_inputs:
                    self.memory_core.store_memory(
                        text=detected_text,
                        memory_type="episodic",
                        speaker="user"
                    )

                # Reset think tag filter state for new request
                self._reset_think_filter_state()

                # Prepare messages with optional memory context
                messages = self.conversation_history.copy()

                # Insert memory context before the last user message if available
                if memory_context:
                    # Insert memory context as a system message before the current user input
                    memory_message = {"role": "system", "content": memory_context}
                    # Insert before the last message (current user input)
                    messages.insert(-1, memory_message)

                data = {
                    "model": self.model_name,
                    "stream": True,
                    "messages": messages,
                    # Add other parameters like temperature, max_tokens if needed from config
                }

                # Debug logging to check for duplicate messages
                logger.debug(f"LLM Processor: Conversation history length: {len(self.conversation_history)}")
                if len(self.conversation_history) <= 5:  # Only log short conversations to avoid spam
                    logger.debug(f"LLM Processor: Full conversation history: {self.conversation_history}")
                else:
                    logger.debug(f"LLM Processor: Last 3 messages: {self.conversation_history[-3:]}")

                sentence_buffer: list[str] = []
                assistant_response_buffer: list[str] = []  # Track full assistant response
                try:
                    logger.debug(f"LLM Processor: Sending POST to {self.completion_url}")
                    with requests.post(
                        str(self.completion_url),
                        headers=self.prompt_headers,
                        json=data,
                        stream=True,
                        timeout=30,  # Add a timeout for the request itself
                    ) as response:
                        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                        logger.debug("LLM Processor: Request to LLM successful, processing stream...")
                        for line in response.iter_lines():
                            if not self.processing_active_event.is_set() or self.shutdown_event.is_set():
                                logger.info("LLM Processor: Interruption or shutdown detected during LLM stream.")
                                break  # Stop processing stream

                            if line:
                                cleaned_line_data = self._clean_raw_bytes(line)
                                if cleaned_line_data:
                                    chunk = self._process_chunk(cleaned_line_data)
                                    if chunk:  # Chunk can be an empty string, but None means no actual content
                                        sentence_buffer.append(chunk)
                                        assistant_response_buffer.append(chunk)  # Track for conversation history
                                        # Split on defined punctuation or if chunk itself is punctuation
                                        if chunk.strip() in self.PUNCTUATION_SET and (
                                            len(sentence_buffer) < 2 or not sentence_buffer[-2].strip().isdigit()
                                        ):
                                            self._process_sentence_for_tts(sentence_buffer)
                                            sentence_buffer = []
                                    # OpenAI [DONE]
                                    elif cleaned_line_data.get("done_marker"):  # OpenAI [DONE]
                                        break
                                    # Ollama end
                                    elif cleaned_line_data.get("done") and cleaned_line_data.get("response") == "":
                                        break

                        # After loop, process any remaining buffer content if not interrupted
                        if self.processing_active_event.is_set() and sentence_buffer:
                            self._process_sentence_for_tts(sentence_buffer)

                        # Add the complete assistant response to conversation history
                        # Only add if we completed normally (not interrupted)
                        if assistant_response_buffer and self.processing_active_event.is_set():
                            full_response = "".join(assistant_response_buffer).strip()
                            if full_response:  # Only add non-empty responses
                                self.conversation_history.append({"role": "assistant", "content": full_response})
                                logger.debug(f"LLM Processor: Added assistant response to history: '{full_response[:100]}...'")

                                # Store assistant response in memory if enabled
                                if self.memory_core and self.store_assistant_responses:
                                    self.memory_core.store_memory(
                                        text=full_response,
                                        memory_type="episodic",
                                        speaker="assistant"
                                    )
                            else:
                                logger.debug("LLM Processor: Empty assistant response, not adding to history")
                        elif assistant_response_buffer and not self.processing_active_event.is_set():
                            logger.debug("LLM Processor: Response was interrupted, not adding partial response to history")

                except requests.exceptions.ConnectionError as e:
                    logger.error(f"LLM Processor: Connection error to LLM service: {e}")
                    self.tts_input_queue.put(
                        "I'm unable to connect to my thinking module. Please check the LLM service connection."
                    )
                except requests.exceptions.Timeout as e:
                    logger.error(f"LLM Processor: Request to LLM timed out: {e}")
                    self.tts_input_queue.put("My brain seems to be taking too long to respond. It might be overloaded.")
                except requests.exceptions.HTTPError as e:
                    status_code = (
                        e.response.status_code
                        if hasattr(e, "response") and hasattr(e.response, "status_code")
                        else "unknown"
                    )
                    logger.error(f"LLM Processor: HTTP error {status_code} from LLM service: {e}")
                    self.tts_input_queue.put(f"I received an error from my thinking module. HTTP status {status_code}.")
                except requests.exceptions.RequestException as e:
                    logger.error(f"LLM Processor: Request to LLM failed: {e}")
                    self.tts_input_queue.put("Sorry, I encountered an error trying to reach my brain.")
                except Exception as e:
                    logger.exception(f"LLM Processor: Unexpected error during LLM request/streaming: {e}")
                    self.tts_input_queue.put("I'm having a little trouble thinking right now.")
                    # Reset filter state on error to prevent stale state
                    self._reset_think_filter_state()
                finally:
                    # Always send EOS if we started processing, unless interrupted early
                    if self.processing_active_event.is_set():  # Only send EOS if not interrupted
                        logger.debug("LLM Processor: Sending EOS token to TTS queue.")
                        self.tts_input_queue.put("<EOS>")
                    else:
                        logger.info("LLM Processor: Interrupted, not sending EOS from LLM processing.")
                        # The AudioPlayer will handle clearing its state.
                        # If an EOS was already sent by TTS from a *previous* partial sentence,
                        # this could lead to an early clear of currently_speaking.
                        # The `processing_active_event` is key to synchronize.

            except queue.Empty:
                pass  # Normal
            except Exception as e:
                logger.exception(f"LLM Processor: Unexpected error in main run loop: {e}")
                time.sleep(0.1)
        logger.info("LanguageModelProcessor thread finished.")
