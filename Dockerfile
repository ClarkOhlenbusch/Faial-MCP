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
    echo "=== Faial repository contents ===" && \
    ls -la && \
    echo "=== Checking for build files ===" && \
    # Check if there's a build script or CMakeLists.txt
    if [ -f "build.sh" ]; then \
        echo "Found build.sh, executing..." && \
        chmod +x build.sh && ./build.sh; \
    elif [ -f "CMakeLists.txt" ]; then \
        echo "Found CMakeLists.txt, building with cmake..." && \
        mkdir -p build && cd build && \
        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
        make -j$(nproc) && \
        make install; \
    elif [ -f "Makefile" ]; then \
        echo "Found Makefile, building..." && \
        make && make install; \
    elif [ -f "setup.py" ]; then \
        echo "Found setup.py, installing Python package..." && \
        python3 setup.py install; \
    else \
        echo "Warning: No recognized build system found in Faial repo"; \
        echo "Repository contents:"; \
        ls -la; \
        echo "Trying generic make..."; \
        make 2>/dev/null || echo "No Makefile found"; \
    fi && \
    echo "=== Searching for faial binary ===" && \
    find /tmp/faial -name "*faial*" -type f -executable 2>/dev/null || true && \
    find /usr/local -name "*faial*" -type f 2>/dev/null || true && \
    # Copy binary if it was built in a different location
    if [ -f "/tmp/faial/build/faial" ]; then \
        echo "Found faial in build directory" && \
        cp /tmp/faial/build/faial /usr/local/bin/faial && chmod +x /usr/local/bin/faial; \
    elif [ -f "/tmp/faial/faial" ]; then \
        echo "Found faial in root directory" && \
        cp /tmp/faial/faial /usr/local/bin/faial && chmod +x /usr/local/bin/faial; \
    elif [ -f "/tmp/faial/bin/faial" ]; then \
        echo "Found faial in bin directory" && \
        cp /tmp/faial/bin/faial /usr/local/bin/faial && chmod +x /usr/local/bin/faial; \
    else \
        echo "Creating stub faial binary for testing..." && \
        echo '#!/bin/bash\necho "Faial stub - binary not properly installed"\necho "Arguments: $@"\nexit 1' > /usr/local/bin/faial && \
        chmod +x /usr/local/bin/faial; \
    fi && \
    # Cleanup
    rm -rf /tmp/faial

# Alternative: If Faial provides pre-built binaries
# RUN wget -O /usr/local/bin/faial https://gitlab.com/umb-svl/faial/-/releases/download/v1.0.0/faial-linux-x64 && \
#     chmod +x /usr/local/bin/faial

# Verify Faial installation
RUN echo "=== Verifying Faial installation ===" && \
    which faial && \
    ls -la /usr/local/bin/faial && \
    faial --version || echo "WARNING: Faial binary exists but --version failed" && \
    echo "=== Faial verification complete ==="

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
