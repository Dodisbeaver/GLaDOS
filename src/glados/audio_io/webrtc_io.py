import asyncio
import json
import queue
import threading
from typing import Any

import numpy as np
import websockets
from loguru import logger
from numpy.typing import NDArray
from websockets.server import WebSocketServerProtocol

from . import VAD


class WebRTCAudioIO:
    """Audio I/O implementation using WebRTC through WebSocket proxy.

    This class provides an implementation of the AudioIO interface that connects
    as a client to an audio proxy server. The proxy handles WebRTC connections
    from browsers and forwards audio data via WebSocket.
    """

    SAMPLE_RATE: int = 16000  # Sample rate for audio processing
    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    VAD_THRESHOLD: float = 0.8  # Threshold for VAD detection

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
        self._sample_queue: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue()
        self._proxy_url = proxy_url
        self._websocket = None
        self._is_playing = False
        self._client_task = None
        self._client_loop = None
        self._stop_event = threading.Event()
        self._connected = False

    async def _client_handler(self) -> None:
        """Handle WebSocket client connection to audio proxy."""
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self._proxy_url) as websocket:
                    self._websocket = websocket
                    self._connected = True
                    logger.info(f"Connected to audio proxy: {self._proxy_url}")

                    # Send ready signal
                    await websocket.send(json.dumps({
                        "type": "glados_ready",
                        "sample_rate": self.SAMPLE_RATE
                    }))

                    # Keep connection alive until stop event
                    while not self._stop_event.is_set():
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            data = json.loads(message)
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
                    logger.info("Retrying connection in 5 seconds...")
                    await asyncio.sleep(5)
            finally:
                self._websocket = None
                self._connected = False

        logger.info("WebSocket client handler stopped")

    async def _process_websocket_message(self, data: dict[str, Any]) -> None:
        """Process incoming WebSocket messages from the audio proxy."""
        message_type = data.get("type")

        if message_type == "audio_data":
            # Receive audio data from WebRTC client via proxy
            audio_samples = np.array(data["samples"], dtype=np.float32)

            # Apply VAD to the received audio
            vad_value = self._vad_model(np.expand_dims(audio_samples, 0))
            vad_confidence = vad_value > self.vad_threshold

            # Put audio sample in queue for GLaDOS processing
            self._sample_queue.put((audio_samples, bool(vad_confidence)))

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

        # Send audio data to all connected WebRTC clients
        message = {
            "type": "audio_playback",
            "samples": audio_data.tolist(),
            "sample_rate": sample_rate,
            "text": text
        }

        # Send to audio proxy using thread-safe approach
        if self._connected and self._client_loop is not None:
            async def send_to_proxy():
                try:
                    if self._websocket is not None:
                        await self._websocket.send(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to send audio to proxy: {e}")

            # Use thread-safe scheduling to the client's event loop
            try:
                future = asyncio.run_coroutine_threadsafe(send_to_proxy(), self._client_loop)
                # Don't wait for completion to avoid blocking
            except Exception as e:
                logger.error(f"Failed to schedule audio send: {e}")
        else:
            logger.warning("Not connected to audio proxy for sending audio")

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
                    future = asyncio.run_coroutine_threadsafe(send_stop_to_proxy(), self._client_loop)
                    # Don't wait for completion to avoid blocking
                except Exception as e:
                    logger.error(f"Failed to schedule stop message: {e}")
            else:
                logger.warning("Not connected to audio proxy for sending stop message")

    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        """Get the queue containing audio samples and VAD confidence."""
        return self._sample_queue