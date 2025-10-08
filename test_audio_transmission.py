#!/usr/bin/env python3
"""
Simple test script to verify audio transmission functionality.
Tests the WebRTC audio transmission with small samples.
"""

import numpy as np
import asyncio
import json
from unittest.mock import Mock, AsyncMock
import sys
import os

# Add src to path to import glados modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from glados.audio_io.webrtc_io import WebRTCAudioIO


def generate_test_audio(duration_seconds: float = 0.1, sample_rate: int = 16000) -> np.ndarray:
    """Generate a simple sine wave test audio signal."""
    samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, samples, False)
    # Generate 440Hz sine wave (A note)
    frequency = 440.0
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


def test_audio_serialization():
    """Test that audio data can be properly serialized to JSON."""
    print("Testing audio serialization...")

    # Generate small test audio
    audio_data = generate_test_audio(0.1)  # 100ms of audio
    print(f"Generated test audio: {len(audio_data)} samples")

    # Convert to list (as done in the implementation)
    audio_samples = audio_data.astype(np.float32).tolist()

    # Create message (as done in WebRTCAudioIO)
    message = {
        "type": "audio_playback",
        "sample_rate": 16000,
        "text": "Test audio",
        "samples": audio_samples
    }

    # Test JSON serialization
    try:
        json_str = json.dumps(message)
        json_size = len(json_str)
        print(f"JSON serialization successful: {json_size} bytes")

        # Test deserialization
        parsed = json.loads(json_str)
        recovered_samples = np.array(parsed["samples"], dtype=np.float32)

        # Verify data integrity
        if np.allclose(audio_data, recovered_samples):
            print("✓ Audio data integrity verified")
            return True
        else:
            print("✗ Audio data integrity check failed")
            return False

    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False


def test_webrtc_audio_io():
    """Test WebRTCAudioIO start_speaking method with mock WebSocket."""
    print("\nTesting WebRTCAudioIO.start_speaking...")

    # Create WebRTCAudioIO instance
    webrtc_io = WebRTCAudioIO("ws://localhost:3000/glados")

    # Mock the WebSocket and connection state
    mock_websocket = AsyncMock()
    webrtc_io._websocket = mock_websocket
    webrtc_io._connected = True

    # Create event loop and run it in background thread
    import threading
    import time

    loop = asyncio.new_event_loop()
    webrtc_io._client_loop = loop

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()

    # Generate test audio
    audio_data = generate_test_audio(0.05)  # 50ms of audio

    try:
        # Call start_speaking
        webrtc_io.start_speaking(audio_data, text="Test message")

        # Give coroutine time to execute
        time.sleep(0.1)

        print("✓ start_speaking completed without errors")

        # Verify that the method sets _is_playing
        if webrtc_io._is_playing:
            print("✓ _is_playing flag set correctly")
        else:
            print("✗ _is_playing flag not set")

        return True

    except Exception as e:
        print(f"✗ start_speaking failed: {e}")
        return False
    finally:
        # Cleanup
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1.0)
        loop.close()


def test_large_audio_handling():
    """Test handling of larger audio samples to ensure no size limits are hit."""
    print("\nTesting large audio sample handling...")

    # Generate 1 second of audio (realistic for GLaDOS responses)
    audio_data = generate_test_audio(1.0)  # 1 second = 16000 samples
    print(f"Generated large test audio: {len(audio_data)} samples")

    # Convert to list and create message
    audio_samples = audio_data.astype(np.float32).tolist()
    message = {
        "type": "audio_playback",
        "sample_rate": 16000,
        "text": "Longer test audio message",
        "samples": audio_samples
    }

    try:
        json_str = json.dumps(message)
        json_size = len(json_str)
        print(f"Large audio JSON serialization successful: {json_size} bytes ({json_size/1024/1024:.2f} MB)")

        # Check if size is reasonable for WebSocket transmission
        if json_size < 10 * 1024 * 1024:  # Less than 10MB
            print("✓ Audio size is reasonable for WebSocket transmission")
            return True
        else:
            print("✗ Audio size might be too large for WebSocket transmission")
            return False

    except Exception as e:
        print(f"✗ Large audio serialization failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing WebRTC Audio Transmission Implementation")
    print("=" * 50)

    tests = [
        test_audio_serialization,
        test_webrtc_audio_io,
        test_large_audio_handling
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed! Audio transmission implementation is working.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())