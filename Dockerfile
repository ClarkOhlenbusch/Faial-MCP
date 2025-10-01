# Use Ubuntu 24.04 as base (same as Faial binary distribution)
FROM ubuntu:24.04

# Install Python 3.10+ and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv curl && \
    rm -rf /var/lib/apt/lists/*

# Download and extract Faial binaries
ADD https://gitlab.com/umb-svl/faial/-/jobs/artifacts/main/raw/bundle/faial.tar.bz2?job=bundle-lin&inline=false /tmp/faial.tar.bz2
RUN mkdir -p /opt/faial && \
    tar -xjf /tmp/faial.tar.bz2 -C /opt/faial && \
    rm /tmp/faial.tar.bz2

# Add Faial binaries to PATH
ENV PATH="/opt/faial:$PATH"

# Set working directory for the MCP server
WORKDIR /app

# Copy the project files
COPY pyproject.toml .
COPY faial_mcp_server/ ./faial_mcp_server/
COPY README.md .

# Install the MCP server package
# Use --break-system-packages because this is an isolated container (PEP 668)
RUN pip3 install --no-cache-dir --break-system-packages -e .

# Set environment variable to point to Faial executable
ENV FAIAL_MCP_EXECUTABLE=/opt/faial/faial-drf

# Expose port for SSE/HTTP transports (optional, for network-based MCP)
EXPOSE 8000

# Default to SSE transport for network-based access
# Can be overridden with: docker run ... faial-mcp --transport stdio
CMD ["faial-mcp-server", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
