FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
  libportaudio2 \
  portaudio19-dev \
  espeak-ng \
  git \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency metadata first so that dependency layers cache well
COPY pyproject.toml README.md ./

# Install project dependencies; cache download artifacts between builds
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --extra api --extra cpu --no-dev

# Copy the rest of the application source
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./src_models/

# Runtime entrypoint handles first-run model download into a persisted volume
COPY docker/glados-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /app/models /tmp && chmod 1777 /tmp
VOLUME ["/app/models"]

ENV GLADOS_CONFIG_PATH=/app/configs/glados_config.yaml
ENV TMPDIR=/tmp
ENV GLADOS_MODELS_PATH=/app/models
ENV GLADOS_AUDIO_IO=webrtc

EXPOSE 5050

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uv", "run", "litestar", "--app", "glados.api.app:app", "run", "--host", "0.0.0.0", "--port", "5050"]
