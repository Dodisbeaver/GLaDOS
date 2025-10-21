"""
Piper TTS implementation using standard Piper library.
This replaces the custom ONNX implementation with the more robust Piper setup from PotatOS.
"""

import json
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
import piper

from ..utils.resources import resource_path


class SpeechSynthesizerProtocol(Protocol):
    """Protocol for speech synthesizers."""

    sample_rate: int

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """Generate speech audio from text."""
        ...


class PiperTTSSynthesizer:
    """
    Piper TTS synthesizer using the standard Piper library.

    This implementation uses the GLaDOS voice model from PotatOS for better
    audio quality and simpler integration.
    """

    # Default paths for Piper models
    MODEL_PATH = resource_path("models/TTS/glados_piper_medium.onnx")
    CONFIG_PATH = resource_path("models/TTS/glados_piper_medium.onnx.json")

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        config_path: Path = CONFIG_PATH,
        speaker_id: int | None = None
    ) -> None:
        """
        Initialize the Piper TTS synthesizer.

        Args:
            model_path: Path to the Piper ONNX model file
            config_path: Path to the Piper JSON configuration file
            speaker_id: Optional speaker ID for multi-speaker models
        """
        # Try multiple possible paths for the Piper models
        import os
        models_path = os.getenv("GLADOS_MODELS_PATH", "/app/models")

        possible_model_paths = [
            model_path,
            Path(f"{models_path}/TTS/glados_piper_medium.onnx"),
            Path("/app/models/TTS/glados_piper_medium.onnx"),
            Path("./models/TTS/glados_piper_medium.onnx")
        ]

        possible_config_paths = [
            config_path,
            Path(f"{models_path}/TTS/glados_piper_medium.onnx.json"),
            Path("/app/models/TTS/glados_piper_medium.onnx.json"),
            Path("./models/TTS/glados_piper_medium.onnx.json")
        ]

        # Find existing model files
        self.model_path = None
        self.config_path = None

        for path in possible_model_paths:
            if path.exists():
                self.model_path = path
                break

        for path in possible_config_paths:
            if path.exists():
                self.config_path = path
                break

        if self.model_path is None:
            raise RuntimeError(f"Piper model not found. Tried: {[str(p) for p in possible_model_paths]}")

        if self.config_path is None:
            raise RuntimeError(f"Piper config not found. Tried: {[str(p) for p in possible_config_paths]}")

        self.speaker_id = speaker_id

        # Load configuration to get sample rate
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.sample_rate = self.config["audio"]["sample_rate"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load Piper configuration from {self.config_path}: {e}")

        # Initialize Piper voice
        try:
            self.voice = piper.PiperVoice.load(str(self.model_path), str(self.config_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load Piper voice from {self.model_path}: {e}")

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """
        Generate speech audio from input text using Piper TTS.

        Args:
            text: The text to synthesize

        Returns:
            Audio data as numpy array of float32 samples
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32)

        try:
            # Synthesize audio using Piper - returns a generator of audio chunks
            audio_generator = self.voice.synthesize(text)

            # Collect all audio chunks from the generator
            audio_arrays = []
            for chunk in audio_generator:
                # Extract the audio data from AudioChunk object
                if hasattr(chunk, 'audio_float_array'):
                    audio_arrays.append(chunk.audio_float_array)
                elif hasattr(chunk, 'audio'):
                    audio_arrays.append(chunk.audio)
                else:
                    # If it's already a numpy array
                    audio_arrays.append(chunk)

            # Concatenate all chunks into a single audio array
            if audio_arrays:
                audio_data = np.concatenate(audio_arrays, axis=0)
            else:
                audio_data = np.array([], dtype=np.float32)

            # Ensure float32 format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            return audio_data

        except Exception as e:
            raise RuntimeError(f"Failed to synthesize speech: {e}")

    def __del__(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'voice'):
            del self.voice