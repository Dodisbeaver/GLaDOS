FROM node:20-slim

# Install curl for health checks, git for GitHub packages, and build tools for native modules
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install --only=production

# Copy application code
COPY . .

# Create public directory for web interface
RUN mkdir -p public

# Fix permissions for node user
RUN chown -R node:node /app && chmod -R 755 public

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Run as non-root user
USER node

CMD ["npm", "start"]
