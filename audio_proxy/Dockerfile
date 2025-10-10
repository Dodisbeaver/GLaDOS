FROM node:20-slim

# Install curl for health checks, git for GitHub packages, build tools for native modules, and openssl for SSL certificates
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3 \
    openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install --only=production

# Copy application code
COPY . .

# Create public directory for web interface and ssl directory
RUN mkdir -p public ssl

# Make scripts executable
RUN chmod +x generate-ssl.sh entrypoint.sh

# Fix permissions for node user
RUN chown -R node:node /app && chmod -R 755 public ssl

# Expose ports
EXPOSE 6080
EXPOSE 6443

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:6080/health || exit 1

# Run as non-root user
USER node

# Use entrypoint script to handle SSL certificate generation
CMD ["./entrypoint.sh"]
