import asyncio
import json
import threading
from typing import Any, AsyncGenerator

import numpy as np
import websockets
from loguru import logger
from numpy.typing import NDArray

from . import VAD


class WebRTCAudioIO:
    """Audio I/O implementation using WebRTC through WebSocket proxy.

    This class provides an implementation of the AudioIO interface that connects
    as a client to an audio proxy server. The proxy handles WebRTC connections
    from browsers and forwards audio data via WebSocket.
    """

    SAMPLE_RATE: int = 16000  # Sample rate for audio processing
    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    VAD_THRESHOLD: float = 0.6  # Threshold for VAD detection (lowered from 0.8 for less sensitivity)
    CHUNK_SIZE: int = 3200  # Samples per chunk (~200ms at 16kHz, ~65KB JSON)

    def __init__(self, proxy_url: str = "ws://audio-proxy:3000/glados", vad_threshold: float | None = None) -> None:
        """Initialize the WebRTC audio I/O.

        Args:
            proxy_url: URL of the audio proxy WebSocket endpoint (default: ws://audio-proxy:3000/glados)
            vad_threshold: Threshold for VAD detection (default: 0.8)
        """
        if vad_threshold is None:
            self.vad_threshold = self.VAD_THRESHOLD
        else:
            self.vad_threshold = vad_threshold

        if not 0 <= self.vad_threshold <= 1:
            raise ValueError("VAD threshold must be between 0 and 1")

        self._vad_model = VAD()
        # Allow a few seconds of headroom so ASR hiccups do not drop audio (~32ms per chunk)
        self._sample_queue: asyncio.Queue[tuple[NDArray[np.float32], bool]] = asyncio.Queue(maxsize=160)
        self._proxy_url = proxy_url
        self._websocket = None
        self._is_playing = False
        self._client_task = None
        self._client_loop = None
        self._stop_event = threading.Event()
        self._connected = False
        self._audio_buffer = np.array([], dtype=np.float32)  # Buffer for VAD processing

    async def _client_handler(self) -> None:
        """Handle WebSocket client connection to audio proxy."""
        while not self._stop_event.is_set():
            try:
                # Use websockets with explicit compression disabled
                async with websockets.connect(
                    self._proxy_url,
                    compression=None,  # Completely disable compression
                    extensions=[],     # No extensions at all
                    max_size=2**20,    # 1MB max message size
                    ping_interval=20,  # Heartbeat every 20s
                    ping_timeout=10    # 10s timeout
                ) as websocket:
                    self._websocket = websocket
                    self._connected = True
                    logger.info(f"Connected to audio proxy: {self._proxy_url}")

                    # Send ready signal
                    ready_message = json.dumps({
                        "type": "glados_ready",
                        "sample_rate": self.SAMPLE_RATE
                    })
                    logger.debug(f"Sending ready message: {ready_message}")
                    await websocket.send(ready_message)

                    # Wait for proxy acknowledgment before proceeding
                    await asyncio.sleep(0.1)

                    # Keep connection alive until stop event
                    while not self._stop_event.is_set():
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            data = json.loads(message)
                            logger.debug(f"Received message: {data}")
                            await self._process_websocket_message(data)
                        except asyncio.TimeoutError:
                            # Check stop event periodically
                            continue
                        except json.JSONDecodeError:
                            logger.error("Invalid JSON received from audio proxy")
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("WebSocket connection closed by proxy")
                            break

            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                if not self._stop_event.is_set():
                    # Exponential backoff to prevent rapid reconnection
                    retry_delay = min(5.0, 1.0 * (1.5 ** getattr(self, '_retry_count', 0)))
                    self._retry_count = getattr(self, '_retry_count', 0) + 1
                    logger.info(f"Retrying connection in {retry_delay:.1f} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    break
            finally:
                self._websocket = None
                self._connected = False

        logger.info("WebSocket client handler stopped")

    async def _process_websocket_message(self, data: dict[str, Any]) -> None:
        """Process incoming WebSocket messages from the audio proxy."""
        message_type = data.get("type")

        if message_type == "audio_data":
            # Receive audio data from WebRTC client via proxy
            incoming_samples = np.array(data["samples"], dtype=np.float32)

            # Add to buffer
            self._audio_buffer = np.concatenate([self._audio_buffer, incoming_samples])

            # Process in 512-sample chunks (required by VAD model for 16kHz)
            vad_chunk_size = 512
            while len(self._audio_buffer) >= vad_chunk_size:
                # Extract chunk
                audio_chunk = self._audio_buffer[:vad_chunk_size]
                self._audio_buffer = self._audio_buffer[vad_chunk_size:]

                # Apply VAD to the chunk
                try:
                    vad_value = self._vad_model(np.expand_dims(audio_chunk, 0))
                    vad_confidence = vad_value > self.vad_threshold
                except Exception as e:
                    logger.warning(f"VAD processing failed: {e}, skipping chunk")
                    vad_confidence = False

                # Put audio chunk in queue with backpressure handling
                try:
                    self._sample_queue.put_nowait((audio_chunk, bool(vad_confidence)))
                except asyncio.QueueFull:
                    logger.warning("Audio queue full, dropping sample to prevent backpressure")

        elif message_type == "proxy_ready":
            # Audio proxy is ready
            logger.info(f"Audio proxy ready with {data.get('clients_connected', 0)} clients")

        elif message_type == "client_connected":
            # New WebRTC client connected via proxy
            logger.info(f"WebRTC client connected (total: {data.get('clients_total', 0)})")

        elif message_type == "client_disconnected":
            # WebRTC client disconnected via proxy
            logger.info(f"WebRTC client disconnected (total: {data.get('clients_total', 0)})")

    def start_listening(self) -> None:
        """Start the WebSocket client connection to audio proxy."""
        if self._client_task is not None:
            self.stop_listening()

        # Clear the stop event for a fresh start
        self._stop_event.clear()

        # Run the client in a separate thread with dedicated event loop
        def client_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._client_loop = loop  # Store reference for thread-safe operations
            try:
                loop.run_until_complete(self._client_handler())
            finally:
                loop.close()
                self._client_loop = None

        self._client_task = threading.Thread(target=client_thread, daemon=True)
        self._client_task.start()

    def stop_listening(self) -> None:
        """Stop the WebSocket client and clean up resources."""
        logger.info("Stopping WebSocket client...")
        self._stop_event.set()

        # Wait for client thread to finish
        if self._client_task is not None:
            self._client_task.join(timeout=10.0)
            if self._client_task.is_alive():
                logger.warning("Client thread did not stop cleanly")
            self._client_task = None

        self._client_loop = None

    def start_speaking(self, audio_data: NDArray[np.float32], sample_rate: int | None = None, text: str = "") -> None:
        """Send audio data to WebRTC clients for playback.

        Audio is automatically chunked to avoid WebSocket size limits.
        Large audio is split into ~200ms chunks for smooth streaming.

        Parameters:
            audio_data: The audio data to play as a numpy float32 array
            sample_rate: The sample rate of the audio data in Hz
            text: Optional text associated with the audio
        """
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
            raise ValueError("Invalid audio data")

        if sample_rate is None:
            sample_rate = self.SAMPLE_RATE

        # Stop any existing playback
        self.stop_speaking()

        self._is_playing = True

        total_samples = len(audio_data)
        logger.info(f"Sending audio: {total_samples} samples ({total_samples/sample_rate:.2f}s), text: '{text}'")

        # Send to audio proxy using thread-safe approach
        if self._connected and self._client_loop is not None and self._websocket is not None:
            async def send_chunked_audio():
                try:
                    if self._websocket is None:
                        logger.warning("WebSocket is closed, cannot send audio")
                        return

                    # Calculate number of chunks needed
                    num_chunks = (total_samples + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE

                    # Send start message with metadata
                    start_message = {
                        "type": "audio_start",
                        "sample_rate": sample_rate,
                        "text": text,
                        "total_samples": total_samples,
                        "chunk_size": self.CHUNK_SIZE,
                        "num_chunks": num_chunks
                    }
                    await self._websocket.send(json.dumps(start_message))
                    logger.debug(f"Starting chunked audio: {num_chunks} chunks")

                    all_chunks_sent = True
                    # Send audio in chunks
                    for chunk_idx in range(num_chunks):
                        if not self._is_playing:
                            logger.info("Playback stopped, cancelling remaining chunks")
                            all_chunks_sent = False
                            break

                        start_idx = chunk_idx * self.CHUNK_SIZE
                        end_idx = min(start_idx + self.CHUNK_SIZE, total_samples)
                        chunk = audio_data[start_idx:end_idx]

                        chunk_message = {
                            "type": "audio_chunk",
                            "chunk_index": chunk_idx,
                            "samples": chunk.astype(np.float32).tolist()
                        }

                        chunk_json = json.dumps(chunk_message)
                        await self._websocket.send(chunk_json)
                        logger.debug(f"Sent chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_json)} bytes)")

                        # Small delay between chunks for flow control (~10ms)
                        await asyncio.sleep(0.01)

                    if all_chunks_sent and self._is_playing:
                        end_message = {"type": "audio_end"}
                        await self._websocket.send(json.dumps(end_message))
                        logger.debug("Audio transmission complete")

                except Exception as e:
                    logger.error(f"Failed to send chunked audio to proxy: {e}")
                finally:
                    self._is_playing = False

            # Use thread-safe scheduling to the client's event loop
            try:
                asyncio.run_coroutine_threadsafe(send_chunked_audio(), self._client_loop)
            except Exception as e:
                logger.error(f"Failed to schedule audio send: {e}")
        else:
            logger.debug("Not connected to audio proxy for sending audio")

    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        """Monitor audio playback progress.

        For WebRTC implementation, this is simplified as the actual playback
        happens on the client side.

        Args:
            total_samples: Total number of samples in the audio data
            sample_rate: Sample rate of the audio

        Returns:
            tuple[bool, int]: (interrupted, percentage_played)
        """
        if sample_rate is None:
            sample_rate = self.SAMPLE_RATE

        # Calculate expected duration
        duration = total_samples / sample_rate

        # Simple timing-based approach for WebRTC
        import time
        start_time = time.time()

        while self._is_playing and (time.time() - start_time) < duration:
            time.sleep(0.1)

        elapsed = time.time() - start_time
        percentage = min(int((elapsed / duration) * 100), 100)
        interrupted = not self._is_playing

        return interrupted, percentage

    def check_if_speaking(self) -> bool:
        """Check if audio is currently being played."""
        return self._is_playing

    def stop_speaking(self) -> None:
        """Stop audio playback."""
        if self._is_playing:
            self._is_playing = False

            # Send stop message to audio proxy using thread-safe approach
            if self._connected and self._client_loop is not None:
                stop_message = {"type": "stop_playback"}

                async def send_stop_to_proxy():
                    try:
                        if self._websocket is not None:
                            await self._websocket.send(json.dumps(stop_message))
                    except Exception as e:
                        logger.error(f"Failed to send stop message to proxy: {e}")

                # Use thread-safe scheduling to the client's event loop
                try:
                    asyncio.run_coroutine_threadsafe(send_stop_to_proxy(), self._client_loop)
                except Exception as e:
                    logger.error(f"Failed to schedule stop message: {e}")
            else:
                logger.warning("Not connected to audio proxy for sending stop message")

    def get_sample_queue(self) -> asyncio.Queue[tuple[NDArray[np.float32], bool]]:
        """Get the queue containing audio samples and VAD confidence.

        Note: This is now an asyncio.Queue. Use async for consumption.
        For synchronous access from threads, use get_sample_sync() instead.
        """
        return self._sample_queue

    def get_sample_sync(self, timeout: float = 1.0) -> tuple[NDArray[np.float32], bool]:
        """Get audio sample synchronously (for thread-based consumers).

        Args:
            timeout: Timeout in seconds

        Returns:
            tuple[NDArray[np.float32], bool]: Audio samples and VAD confidence

        Raises:
            asyncio.QueueEmpty: If no sample available within timeout
        """
        # Use an event to wait for loop without blocking
        loop_ready_event = threading.Event()

        def check_loop_ready():
            """Non-blocking check for event loop readiness."""
            max_attempts = 20  # 1 second total at 0.05s intervals
            for _ in range(max_attempts):
                if self._client_loop is not None:
                    loop_ready_event.set()
                    return
                threading.Event().wait(0.05)  # Non-blocking sleep alternative
            # Timeout reached, event remains unset

        # Start loop check in a separate thread to avoid blocking
        check_thread = threading.Thread(target=check_loop_ready, daemon=True)
        check_thread.start()

        # Wait for loop to be ready or timeout
        if not loop_ready_event.wait(timeout=1.0):
            # Event loop not ready within timeout
            raise asyncio.QueueEmpty("Event loop not ready")

        # Double-check loop is still available (race condition protection)
        current_loop = self._client_loop
        if current_loop is None:
            raise asyncio.QueueEmpty("Event loop became unavailable")

        # Schedule the async get on the event loop and wait for result
        try:
            future = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(self._sample_queue.get(), timeout=timeout),
                current_loop
            )
            result = future.result(timeout=timeout + 0.5)  # Extra time for scheduling
            return result
        except TimeoutError:
            raise asyncio.QueueEmpty("Timeout waiting for audio sample")
        except RuntimeError as e:
            # Handle case where event loop is closed/unavailable
            raise asyncio.QueueEmpty(f"Event loop error: {e}")

    async def stream_audio_samples(self) -> AsyncGenerator[tuple[NDArray[np.float32], bool], None]:
        """Async generator that yields audio samples from the queue.

        Yields:
            tuple[NDArray[np.float32], bool]: Audio samples and VAD confidence

        Example:
            async for audio_data, has_voice in webrtc_io.stream_audio_samples():
                process_audio(audio_data, has_voice)
        """
        while True:
            try:
                audio_data, vad_confidence = await self._sample_queue.get()
                yield audio_data, vad_confidence
                self._sample_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Audio streaming cancelled")
                break
            except Exception as e:
                logger.error(f"Error streaming audio samples: {e}")
                break
