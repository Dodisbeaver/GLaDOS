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

    This class provides an implementation of the AudioIO interface that receives
    audio data from a WebRTC-enabled web interface via WebSocket connections.
    It handles real-time audio capture with voice activity detection and audio playback.
    """

    SAMPLE_RATE: int = 16000  # Sample rate for audio processing
    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    VAD_THRESHOLD: float = 0.8  # Threshold for VAD detection

    def __init__(self, websocket_port: int = 8765, vad_threshold: float | None = None) -> None:
        """Initialize the WebRTC audio I/O.

        Args:
            websocket_port: Port for WebSocket server (default: 8765)
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
        self._websocket_port = websocket_port
        self._websocket_server = None
        self._connected_clients: set[WebSocketServerProtocol] = set()
        self._is_playing = False
        self._server_task = None
        self._stop_event = threading.Event()

    async def _handle_websocket(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle incoming WebSocket connections and audio data."""
        self._connected_clients.add(websocket)
        logger.info(f"WebRTC client connected: {websocket.remote_address}")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_websocket_message(data, websocket)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from WebRTC client")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebRTC client disconnected")
        finally:
            self._connected_clients.discard(websocket)

    async def _process_websocket_message(self, data: dict[str, Any], websocket: WebSocketServerProtocol) -> None:
        """Process incoming WebSocket messages from the WebRTC client."""
        message_type = data.get("type")

        if message_type == "audio_data":
            # Receive audio data from WebRTC client
            audio_samples = np.array(data["samples"], dtype=np.float32)

            # Apply VAD to the received audio
            vad_value = self._vad_model(np.expand_dims(audio_samples, 0))
            vad_confidence = vad_value > self.vad_threshold

            # Put audio sample in queue for GLaDOS processing
            self._sample_queue.put((audio_samples, bool(vad_confidence)))

        elif message_type == "client_ready":
            # Client is ready to receive audio
            await websocket.send(json.dumps({
                "type": "server_ready",
                "sample_rate": self.SAMPLE_RATE
            }))

    def start_listening(self) -> None:
        """Start the WebSocket server for receiving audio from WebRTC clients."""
        if self._server_task is not None:
            self.stop_listening()

        async def run_server():
            self._websocket_server = await websockets.serve(
                self._handle_websocket,
                "0.0.0.0",
                self._websocket_port
            )
            logger.info(f"WebRTC WebSocket server started on port {self._websocket_port}")
            await self._websocket_server.wait_closed()

        # Run the server in a separate thread
        loop = asyncio.new_event_loop()

        def server_thread():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_server())

        self._server_task = threading.Thread(target=server_thread, daemon=True)
        self._server_task.start()

    def stop_listening(self) -> None:
        """Stop the WebSocket server and clean up resources."""
        if self._websocket_server is not None:
            # Close all connected clients
            for client in self._connected_clients.copy():
                asyncio.create_task(client.close())

            # Close the server
            self._websocket_server.close()
            self._websocket_server = None

        if self._server_task is not None:
            self._server_task.join(timeout=5.0)
            self._server_task = None

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

        # Send to all connected clients asynchronously
        for client in self._connected_clients.copy():
            try:
                asyncio.create_task(client.send(json.dumps(message)))
            except Exception as e:
                logger.error(f"Failed to send audio to WebRTC client: {e}")
                self._connected_clients.discard(client)

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

            # Send stop message to all connected clients
            stop_message = {"type": "stop_playback"}
            for client in self._connected_clients.copy():
                try:
                    asyncio.create_task(client.send(json.dumps(stop_message)))
                except Exception as e:
                    logger.error(f"Failed to send stop message to WebRTC client: {e}")

    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        """Get the queue containing audio samples and VAD confidence."""
        return self._sample_queue