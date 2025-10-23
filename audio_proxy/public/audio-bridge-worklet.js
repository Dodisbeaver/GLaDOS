/**
 * AudioWorklet processor for GLaDOS WebRTC audio bridge.
 *
 * This worklet replaces the deprecated ScriptProcessorNode for capturing
 * microphone audio and sending it to the GLaDOS backend via WebSocket.
 *
 * The processor runs in the audio rendering thread for better performance
 * and reduced latency compared to ScriptProcessorNode.
 */

class AudioBridgeProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        // Set up message handling for receiving samples to visualize
        this.port.onmessage = (event) => {
            if (event.data.type === 'configure') {
                // Future: handle configuration if needed
            }
        };
    }

    /**
     * Process audio samples from the microphone input.
     *
     * This is called on the audio rendering thread at regular intervals
     * (128 frames per call by default).
     *
     * @param {Float32Array[][]} inputs - Input audio data (input[channel][sample])
     * @param {Float32Array[][]} outputs - Output audio data (not used)
     * @param {Object} parameters - Audio parameters (not used)
     * @returns {boolean} - true to keep processor alive
     */
    process(inputs, outputs, parameters) {
        // Get the first input (microphone)
        const input = inputs[0];

        if (input && input.length > 0) {
            // Get the first channel (mono audio)
            const inputChannel = input[0];

            if (inputChannel && inputChannel.length > 0) {
                // Send audio data to the main thread for WebSocket transmission
                // We need to copy the data because Float32Arrays from the audio
                // thread are neutered after process() returns
                const audioData = new Float32Array(inputChannel);

                this.port.postMessage({
                    type: 'audio_data',
                    samples: audioData
                });
            }
        }

        // Return true to keep the processor alive
        return true;
    }
}

// Register the processor
registerProcessor('audio-bridge-processor', AudioBridgeProcessor);
