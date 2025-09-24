# Use Node.js 18+ as base image
FROM node:18-slim

# Install system dependencies for running Faial CLI
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY src/ ./src/
COPY tsconfig.json ./

# Build the application
RUN npm run build

# Install Faial CLI
# TODO: Replace this with actual Faial CLI installation
# This is a placeholder - you'll need to update based on how Faial CLI is distributed
ENV FAIAL_PATH=/usr/local/bin/faial

# Option 1: If Faial CLI is distributed as a binary
# RUN wget -O /usr/local/bin/faial https://example.com/faial-cli && chmod +x /usr/local/bin/faial

# Option 2: If Faial CLI is distributed via package manager
# RUN apt-get update && apt-get install -y faial-cli

# Option 3: If Faial CLI is available via npm
# RUN npm install -g faial-cli

# Option 4: If Faial CLI needs to be built from source
# RUN git clone https://github.com/your-org/faial-cli.git /tmp/faial-cli && \
#     cd /tmp/faial-cli && \
#     make && \
#     cp faial /usr/local/bin/faial && \
#     rm -rf /tmp/faial-cli

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port for HTTP server
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Set environment variables
ENV NODE_ENV=production
ENV TRANSPORT_MODE=http
ENV PORT=3000

# Start the server
CMD ["node", "dist/index.js"]
