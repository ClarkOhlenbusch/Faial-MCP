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

# Install all dependencies (including dev dependencies for building)
RUN npm ci

# Copy source code
COPY src/ ./src/
COPY tsconfig.json ./

# Build the application
RUN npm run build

# Create the api directory for Railway deployment
RUN mkdir -p api

# Remove dev dependencies to reduce image size (optional)
# RUN npm prune --production

# Install Faial CLI
# Based on the GitLab repo: https://gitlab.com/umb-svl/faial
ENV FAIAL_PATH=/usr/local/bin/faial

# Install build dependencies for Faial
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone and build Faial from source
RUN git clone https://gitlab.com/umb-svl/faial.git /tmp/faial && \
    cd /tmp/faial && \
    # Check if there's a build script or CMakeLists.txt
    if [ -f "build.sh" ]; then \
        chmod +x build.sh && ./build.sh; \
    elif [ -f "CMakeLists.txt" ]; then \
        mkdir build && cd build && \
        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
        make -j$(nproc) && \
        make install; \
    else \
        echo "Warning: No recognized build system found in Faial repo"; \
        echo "Please check the Faial repository for installation instructions"; \
        echo "and update this Dockerfile accordingly"; \
    fi && \
    # Copy binary if it was built in a different location
    if [ -f "/tmp/faial/build/faial" ]; then \
        cp /tmp/faial/build/faial /usr/local/bin/faial && chmod +x /usr/local/bin/faial; \
    elif [ -f "/tmp/faial/faial" ]; then \
        cp /tmp/faial/faial /usr/local/bin/faial && chmod +x /usr/local/bin/faial; \
    fi && \
    # Cleanup
    rm -rf /tmp/faial

# Alternative: If Faial provides pre-built binaries
# RUN wget -O /usr/local/bin/faial https://gitlab.com/umb-svl/faial/-/releases/download/v1.0.0/faial-linux-x64 && \
#     chmod +x /usr/local/bin/faial

# Verify Faial installation
RUN faial --version || echo "Faial may need manual installation steps - check the repository README"

# Create non-root user for security (but keep root for Railway compatibility)
# RUN useradd -m appuser && chown -R appuser:appuser /app
# USER appuser

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
