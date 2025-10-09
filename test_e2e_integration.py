#!/usr/bin/env python3
"""
End-to-end integration test for Phase 4.
Tests the complete audio pipeline with WebRTC.
"""

import asyncio
import numpy as np
import sys
import os
import time

# Add src to path to import glados modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from glados.audio_io.webrtc_io import WebRTCAudioIO


def generate_test_audio(duration_seconds: float = 0.5, sample_rate: int = 16000) -> np.ndarray:
    """Generate a simple sine wave test audio signal."""
    samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, samples, False)
    # Generate 440Hz sine wave (A note)
    frequency = 440.0
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


async def test_webrtc_connection():
    """Test WebRTC connection to audio proxy."""
    print("Testing WebRTC connection to audio proxy...")

    webrtc_io = WebRTCAudioIO("ws://audio-proxy:3000/glados")

    # Start listening (connects to proxy)
    webrtc_io.start_listening()

    # Wait for connection
    max_wait = 5.0
    start = time.time()
    while not webrtc_io._connected and (time.time() - start) < max_wait:
        await asyncio.sleep(0.1)

    if webrtc_io._connected:
        print("✓ Connected to audio proxy")
        webrtc_io.stop_listening()
        return True
    else:
        print("✗ Failed to connect to audio proxy")
        webrtc_io.stop_listening()
        return False


async def test_chunked_audio_transmission():
    """Test chunked audio transmission through the pipeline."""
    print("\nTesting chunked audio transmission...")

    webrtc_io = WebRTCAudioIO("ws://audio-proxy:3000/glados")
    webrtc_io.start_listening()

    # Wait for connection
    await asyncio.sleep(1.0)

    if not webrtc_io._connected:
        print("✗ Not connected to audio proxy")
        webrtc_io.stop_listening()
        return False

    # Generate test audio that will require chunking (2 seconds)
    audio_data = generate_test_audio(2.0)
    num_samples = len(audio_data)
    num_chunks = (num_samples + webrtc_io.CHUNK_SIZE - 1) // webrtc_io.CHUNK_SIZE

    print(f"Sending {num_samples} samples ({num_samples/16000:.2f}s)")
    print(f"Expected chunks: {num_chunks}")

    try:
        # Send audio
        webrtc_io.start_speaking(audio_data, text="Test audio transmission")

        # Wait for transmission to complete
        await asyncio.sleep(1.0)

        print("✓ Chunked audio sent successfully")

        webrtc_io.stop_listening()
        return True

    except Exception as e:
        print(f"✗ Failed to send chunked audio: {e}")
        webrtc_io.stop_listening()
        return False


async def test_vad_processing():
    """Test Voice Activity Detection processing."""
    print("\nTesting VAD processing...")

    webrtc_io = WebRTCAudioIO("ws://audio-proxy:3000/glados")

    # Test VAD threshold
    if webrtc_io.vad_threshold == 0.8:
        print(f"✓ VAD threshold set to {webrtc_io.vad_threshold}")
    else:
        print(f"✗ Unexpected VAD threshold: {webrtc_io.vad_threshold}")
        return False

    # Verify VAD model is loaded
    if webrtc_io._vad_model is not None:
        print("✓ VAD model loaded")
    else:
        print("✗ VAD model not loaded")
        return False

    # Test VAD on sample audio (must be 512 samples for 16kHz)
    test_audio = generate_test_audio(0.032)  # 32ms = 512 samples at 16kHz

    try:
        # Expand dims for VAD model (expects batch dimension)
        vad_result = webrtc_io._vad_model(np.expand_dims(test_audio, 0))
        print(f"✓ VAD processing works (confidence: {vad_result:.3f})")
        return True
    except Exception as e:
        print(f"✗ VAD processing failed: {e}")
        return False


async def test_audio_queue_flow():
    """Test audio flowing through the queue system."""
    print("\nTesting audio queue flow...")

    webrtc_io = WebRTCAudioIO("ws://audio-proxy:3000/glados")

    # Verify queue is asyncio.Queue
    if isinstance(webrtc_io._sample_queue, asyncio.Queue):
        print("✓ Audio queue is asyncio.Queue")
    else:
        print("✗ Audio queue is wrong type")
        return False

    # Test adding samples to queue
    test_sample = generate_test_audio(0.032)  # 32ms VAD sample

    try:
        webrtc_io._sample_queue.put_nowait((test_sample, True))
        print("✓ Can add samples to queue")

        # Test retrieving samples
        retrieved_sample, vad = await asyncio.wait_for(
            webrtc_io._sample_queue.get(),
            timeout=1.0
        )

        if np.array_equal(test_sample, retrieved_sample) and vad == True:
            print("✓ Can retrieve samples from queue")
            return True
        else:
            print("✗ Sample data mismatch")
            return False

    except Exception as e:
        print(f"✗ Queue operation failed: {e}")
        return False


async def test_async_generator():
    """Test async generator for audio streaming."""
    print("\nTesting async generator streaming...")

    webrtc_io = WebRTCAudioIO("ws://audio-proxy:3000/glados")

    # Add test samples
    test_samples = []
    for i in range(3):
        sample = generate_test_audio(0.032)
        test_samples.append((sample, bool(i % 2)))
        await webrtc_io._sample_queue.put((sample, bool(i % 2)))

    print(f"Added {len(test_samples)} samples to queue")

    # Test async generator consumption
    received = []
    try:
        async def consume():
            async for audio, vad in webrtc_io.stream_audio_samples():
                received.append((audio, vad))
                if len(received) >= 3:
                    break

        await asyncio.wait_for(consume(), timeout=2.0)

        if len(received) == 3:
            print(f"✓ Async generator yielded {len(received)} samples")
            return True
        else:
            print(f"✗ Expected 3 samples, got {len(received)}")
            return False

    except asyncio.TimeoutError:
        print("✗ Async generator timed out")
        return False
    except Exception as e:
        print(f"✗ Async generator failed: {e}")
        return False


async def main():
    """Run all integration tests."""
    print("Phase 4: End-to-End Integration Tests")
    print("=" * 50)

    tests = [
        test_vad_processing,
        test_audio_queue_flow,
        test_async_generator,
        test_webrtc_connection,
        test_chunked_audio_transmission,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All integration tests passed!")
        print("\nThe full pipeline is ready:")
        print("  Microphone → WebRTC → Audio Proxy → GLaDOS → Audio Proxy → WebRTC → Speaker")
        return 0
    else:
        print("✗ Some integration tests failed.")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
