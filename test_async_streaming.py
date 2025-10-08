#!/usr/bin/env python3
"""
Test async streaming patterns from Phase 2.
Verifies asyncio.Queue, async generators, and backpressure handling.
"""

import asyncio
import numpy as np
import sys
import os

# Add src to path to import glados modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from glados.audio_io.webrtc_io import WebRTCAudioIO


async def test_async_queue_operations():
    """Test that the asyncio.Queue works correctly."""
    print("Testing asyncio.Queue operations...")

    webrtc_io = WebRTCAudioIO("ws://localhost:3000/glados")

    # Verify queue is asyncio.Queue
    assert isinstance(webrtc_io._sample_queue, asyncio.Queue), "Queue should be asyncio.Queue"

    # Verify maxsize for backpressure
    assert webrtc_io._sample_queue.maxsize == 10, "Queue maxsize should be 10"

    print("✓ Queue is asyncio.Queue with maxsize=10")
    return True


async def test_backpressure_handling():
    """Test that backpressure handling drops samples when queue is full."""
    print("\nTesting backpressure handling...")

    webrtc_io = WebRTCAudioIO("ws://localhost:3000/glados")

    # Fill the queue to capacity
    for i in range(10):
        audio_data = np.random.random(1024).astype(np.float32)
        webrtc_io._sample_queue.put_nowait((audio_data, True))

    assert webrtc_io._sample_queue.full(), "Queue should be full"
    print(f"✓ Queue filled to capacity: {webrtc_io._sample_queue.qsize()}/10")

    # Simulate receiving audio when queue is full
    # This should trigger backpressure handling
    test_message = {
        "type": "audio_data",
        "samples": np.random.random(512).astype(np.float32).tolist()
    }

    # Should not raise exception due to backpressure handling
    try:
        await webrtc_io._process_websocket_message(test_message)
        print("✓ Backpressure handling works - sample dropped without error")
    except asyncio.QueueFull:
        print("✗ Backpressure handling failed - QueueFull exception raised")
        return False

    return True


async def test_async_generator():
    """Test the async generator pattern for streaming audio."""
    print("\nTesting async generator pattern...")

    webrtc_io = WebRTCAudioIO("ws://localhost:3000/glados")

    # Add some test samples to the queue
    test_samples = []
    for i in range(3):
        audio_data = np.random.random(1024).astype(np.float32)
        vad = bool(i % 2)  # Alternate True/False
        test_samples.append((audio_data, vad))
        await webrtc_io._sample_queue.put((audio_data, vad))

    print(f"✓ Added {len(test_samples)} test samples to queue")

    # Test async generator consumption
    received_samples = []
    async def consume_samples():
        async for audio_data, vad_confidence in webrtc_io.stream_audio_samples():
            received_samples.append((audio_data, vad_confidence))
            if len(received_samples) >= 3:
                break

    # Run consumer with timeout
    try:
        await asyncio.wait_for(consume_samples(), timeout=2.0)
    except asyncio.TimeoutError:
        print("✗ Timeout waiting for samples")
        return False

    # Verify we received all samples
    if len(received_samples) == 3:
        print(f"✓ Async generator yielded {len(received_samples)} samples")

        # Verify data integrity
        for i, (expected, received) in enumerate(zip(test_samples, received_samples)):
            if not np.array_equal(expected[0], received[0]) or expected[1] != received[1]:
                print(f"✗ Sample {i} data mismatch")
                return False

        print("✓ All samples matched expected data")
        return True
    else:
        print(f"✗ Expected 3 samples, got {len(received_samples)}")
        return False


async def test_sync_bridge():
    """Test the synchronous bridge method for backward compatibility."""
    print("\nTesting sync bridge method...")

    webrtc_io = WebRTCAudioIO("ws://localhost:3000/glados")

    # Start event loop in background
    import threading

    loop = asyncio.new_event_loop()
    webrtc_io._client_loop = loop

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()

    try:
        # Add test sample
        test_audio = np.random.random(1024).astype(np.float32)
        await webrtc_io._sample_queue.put((test_audio, True))

        # Test sync get from main thread
        import time
        time.sleep(0.1)  # Give loop time to start

        audio_data, vad = webrtc_io.get_sample_sync(timeout=1.0)

        if np.array_equal(audio_data, test_audio):
            print("✓ Sync bridge method works correctly")
            return True
        else:
            print("✗ Data mismatch in sync bridge")
            return False

    except asyncio.QueueEmpty:
        print("✗ Sync bridge timed out")
        return False

    except Exception as e:
        print(f"✗ Sync bridge failed: {e}")
        return False

    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1.0)
        loop.close()


async def main():
    """Run all async streaming tests."""
    print("Testing Phase 2: Async Streaming Patterns")
    print("=" * 50)

    tests = [
        test_async_queue_operations,
        test_backpressure_handling,
        test_async_generator,
        test_sync_bridge
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
        print("✓ All async streaming tests passed!")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
