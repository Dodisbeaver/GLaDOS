"""
Embedding providers for GLaDOS Memory Core.

This module provides different embedding backends including sentence transformers
and EmbeddingGemma for flexible memory system configuration.
"""

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import requests
from loguru import logger
from sentence_transformers import SentenceTransformer


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode text(s) into embeddings."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name/identifier."""
        pass


class SentenceTransformersProvider(EmbeddingProvider):
    """Standard sentence transformers provider for lightweight models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize sentence transformers provider.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        logger.info(f"Loading sentence transformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.success(f"Loaded {model_name} (dim={self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer {model_name}: {e}")
            raise

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode text(s) into embeddings."""
        try:
            if isinstance(texts, str):
                texts = [texts]
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


class EmbeddingGemmaProvider(EmbeddingProvider):
    """EmbeddingGemma provider for high-quality embeddings."""

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300M",
        prompt_name: str | None = None,
        truncate_dim: int | None = None,
        require_auth: bool = True,
    ):
        """
        Initialize EmbeddingGemma provider.

        Args:
            model_name: EmbeddingGemma model identifier
            prompt_name: Task-specific prompt (e.g., "STS", "RAG")
            truncate_dim: Truncate embeddings to this dimension for efficiency
            require_auth: Whether to require HuggingFace authentication
        """
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.truncate_dim = truncate_dim
        self.require_auth = require_auth

        logger.info(f"Loading EmbeddingGemma model: {model_name}")

        # Check for authentication if required
        if self.require_auth:
            self._ensure_huggingface_auth()

        try:
            # Import here to handle optional dependency
            import torch

            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            # Load model
            self.model = SentenceTransformer(model_name).to(device=self.device)

            # Get embedding dimension (may be truncated)
            base_dim = self.model.get_sentence_embedding_dimension()
            self.embedding_dim = self.truncate_dim if self.truncate_dim else base_dim

            logger.success(
                f"Loaded EmbeddingGemma {model_name} "
                f"(base_dim={base_dim}, output_dim={self.embedding_dim}, device={self.device})"
            )

        except ImportError as e:
            logger.error(f"EmbeddingGemma requires torch: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load EmbeddingGemma {model_name}: {e}")
            logger.info("Make sure you have access to the model and run: huggingface-cli login")
            raise

    def _ensure_huggingface_auth(self) -> None:
        """Ensure HuggingFace authentication for model access."""
        try:
            from huggingface_hub import whoami
            user = whoami()
            logger.debug(f"Authenticated as HuggingFace user: {user['name']}")
        except Exception:
            logger.warning(
                "HuggingFace authentication may be required for EmbeddingGemma. "
                "Run 'huggingface-cli login' if you encounter access errors."
            )

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode text(s) into embeddings."""
        try:
            if isinstance(texts, str):
                texts = [texts]

            # Encode with optional prompt
            if self.prompt_name:
                embeddings = self.model.encode(texts, prompt_name=self.prompt_name)
            else:
                embeddings = self.model.encode(texts)

            # Convert to numpy if needed
            if hasattr(embeddings, "cpu"):
                embeddings = embeddings.cpu().numpy()

            # Truncate if requested
            if self.truncate_dim and embeddings.shape[1] > self.truncate_dim:
                embeddings = embeddings[:, :self.truncate_dim]

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode texts with EmbeddingGemma: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def get_model_name(self) -> str:
        """Get model name."""
        return f"{self.model_name}" + (f"[{self.truncate_dim}d]" if self.truncate_dim else "")


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider for local models including EmbeddingGemma."""

    def __init__(
        self,
        model_name: str = "embeddinggemma",
        ollama_url: str = "http://localhost:11434",
        auto_pull: bool = True,
        max_retries: int = 3,
    ):
        """
        Initialize Ollama embedding provider.

        Args:
            model_name: Ollama model name (e.g., "embeddinggemma", "nomic-embed-text")
            ollama_url: Ollama server URL
            auto_pull: Automatically pull model if not available
            max_retries: Maximum retries for API calls
        """
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.auto_pull = auto_pull
        self.max_retries = max_retries
        self.embedding_dim = None

        logger.info(f"Initializing Ollama embedding provider: {model_name}")

        # Test connection and pull model if needed
        try:
            self._ensure_model_available()
            self._determine_embedding_dimension()
            logger.success(f"Ollama provider ready: {model_name} (dim={self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            raise

    def _ensure_model_available(self) -> None:
        """Ensure the model is available, pulling if necessary."""
        try:
            # Check if model exists
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]

            if self.model_name not in model_names:
                if self.auto_pull:
                    logger.info(f"Model {self.model_name} not found, pulling...")
                    self._pull_model()
                else:
                    raise RuntimeError(f"Model {self.model_name} not available and auto_pull=False")
            else:
                logger.debug(f"Model {self.model_name} is available")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Ollama server at {self.ollama_url}: {e}")

    def _pull_model(self) -> None:
        """Pull the model from Ollama registry."""
        try:
            logger.info(f"Pulling model {self.model_name}...")
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": self.model_name},
                timeout=300,  # 5 minutes for model pull
                stream=True
            )
            response.raise_for_status()

            # Monitor pull progress
            for line in response.iter_lines():
                if line:
                    try:
                        data = line.decode("utf-8")
                        import json
                        status_data = json.loads(data)
                        status = status_data.get("status", "")
                        if "completed" in status_data and "total" in status_data:
                            completed = status_data["completed"]
                            total = status_data["total"]
                            percent = (completed / total) * 100 if total > 0 else 0
                            logger.debug(f"Pull progress: {percent:.1f}%")
                    except (json.JSONDecodeError, KeyError):
                        continue

            logger.success(f"Successfully pulled model {self.model_name}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to pull model {self.model_name}: {e}")

    def _determine_embedding_dimension(self) -> None:
        """Determine the embedding dimension by making a test call."""
        try:
            test_embedding = self.encode("test")
            self.embedding_dim = len(test_embedding[0]) if test_embedding.ndim > 1 else len(test_embedding)
            logger.debug(f"Determined embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            # Default dimensions for known models
            known_dims = {
                "embeddinggemma": 768,
                "nomic-embed-text": 768,
                "mxbai-embed-large": 1024,
                "snowflake-arctic-embed": 768,
                "granite-embedding": 768,
            }
            self.embedding_dim = known_dims.get(self.model_name, 768)
            logger.info(f"Using default dimension {self.embedding_dim} for {self.model_name}")

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode text(s) into embeddings using Ollama."""
        if isinstance(texts, str):
            texts = [texts]

        try:
            embeddings = []
            for text in texts:
                for attempt in range(self.max_retries):
                    try:
                        response = requests.post(
                            f"{self.ollama_url}/api/embeddings",
                            json={
                                "model": self.model_name,
                                "prompt": text
                            },
                            timeout=30
                        )
                        response.raise_for_status()

                        result = response.json()
                        if "embedding" in result:
                            embeddings.append(result["embedding"])
                            break
                        else:
                            raise ValueError(f"No embedding in response: {result}")

                    except requests.exceptions.RequestException as e:
                        if attempt == self.max_retries - 1:
                            raise RuntimeError(f"Failed to get embedding after {self.max_retries} attempts: {e}")
                        logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")

            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            logger.error(f"Failed to encode texts with Ollama: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim or 768

    def get_model_name(self) -> str:
        """Get model name."""
        return f"ollama:{self.model_name}"


def create_embedding_provider(
    provider_type: str = "sentence_transformers",
    model_name: str | None = None,
    **kwargs: Any
) -> EmbeddingProvider:
    """
    Factory function to create embedding providers.

    Args:
        provider_type: Type of provider ("sentence_transformers", "embeddinggemma")
        model_name: Model name/identifier
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured embedding provider

    Raises:
        ValueError: If provider type is not supported
    """
    provider_type = provider_type.lower()

    if provider_type == "sentence_transformers":
        model_name = model_name or "all-MiniLM-L6-v2"
        return SentenceTransformersProvider(model_name=model_name)

    elif provider_type in ("embeddinggemma", "embedding_gemma", "gemma"):
        model_name = model_name or "google/embeddinggemma-300M"
        return EmbeddingGemmaProvider(model_name=model_name, **kwargs)

    elif provider_type in ("ollama", "ollama_embedding"):
        model_name = model_name or "embeddinggemma"
        return OllamaEmbeddingProvider(model_name=model_name, **kwargs)

    else:
        raise ValueError(
            f"Unsupported provider type: {provider_type}. "
            f"Supported types: sentence_transformers, embeddinggemma, ollama"
        )


def get_recommended_provider_for_device() -> tuple[str, str]:
    """
    Get recommended embedding provider and model based on available hardware.

    Returns:
        Tuple of (provider_type, model_name)
    """
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        else:
            gpu_memory = 0
    except ImportError:
        has_cuda = False
        gpu_memory = 0

    # Check if Ollama is available (try to connect)
    try:
        import os
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").replace("/api/chat", "")
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("Ollama server detected, recommending Ollama EmbeddingGemma")
            return "ollama", "embeddinggemma"
    except Exception:
        pass

    # Recommend based on available resources
    if has_cuda and gpu_memory >= 2:  # 2GB+ GPU
        logger.info(f"GPU detected with {gpu_memory:.1f}GB memory, recommending EmbeddingGemma")
        return "embeddinggemma", "google/embeddinggemma-300M"
    else:
        logger.info("CPU or limited GPU detected, recommending lightweight sentence transformer")
        return "sentence_transformers", "all-MiniLM-L6-v2"