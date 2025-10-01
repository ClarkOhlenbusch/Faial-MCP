# Docker Quick Start Guide

This guide shows you how to run Faial MCP Server in Docker without installing Faial locally.

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- (Optional) Docker Compose for easier management

## Quick Start

### Option 1: Docker Run (Simple)

```bash
# Build the image
docker build -t faial-mcp .

# Run the server
docker run -p 8000:8000 faial-mcp
```

Your MCP server is now running at `http://localhost:8000`!

### Option 2: Docker Compose (Recommended)

```bash
# Start the server in the background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the server
docker-compose down
```

## Connecting Your MCP Client

### Claude Desktop

Edit your Claude Desktop config file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add:
```json
{
  "mcpServers": {
    "faial": {
      "url": "http://localhost:8000",
      "transport": "sse"
    }
  }
}
```

Restart Claude Desktop.

### Cursor

In Cursor settings, add an MCP server:
- URL: `http://localhost:8000`
- Transport: `sse`

## Analyzing CUDA Code

### Method 1: Inline Source (Easiest)

Your MCP client can send code directly:

```
Analyze this CUDA kernel for data races:

__global__ void add(int *a, int *b, int n) {
    int i = threadIdx.x;
    if (i < n) {
        a[i] = b[i] + 1;
    }
}
```

### Method 2: Files from Host

1. Place your `.cu` files in the `workspace/` directory
2. Start the container with the workspace mounted (already configured in `docker-compose.yml`)
3. Reference files as `/workspace/your_file.cu`

Example workspace structure:
```
workspace/
├── kernel1.cu
├── kernel2.cu
└── my_project/
    └── main.cu
```

Ask your MCP client: "Analyze `/workspace/kernel1.cu` for data races"

## Advanced Usage

### Custom Timeout

```bash
docker run -p 8000:8000 \
  -e FAIAL_MCP_TIMEOUT_MS=120000 \
  faial-mcp
```

### Different Port

```bash
docker run -p 9000:9000 faial-mcp \
  faial-mcp-server --transport sse --host 0.0.0.0 --port 9000
```

### Mount Custom Directory

```bash
docker run -p 8000:8000 \
  -v /path/to/your/cuda/code:/code \
  faial-mcp
```

Reference files as `/code/your_file.cu`

## Testing the Server

Test with curl:

```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "analyze_kernel",
    "arguments": {
      "source": "__global__ void test() { }",
      "virtual_filename": "test.cu"
    }
  }'
```

## Troubleshooting

### "Connection refused"

- Ensure the container is running: `docker ps`
- Check logs: `docker logs faial-mcp-server` (if using docker-compose)
- Verify port mapping: `docker port faial-mcp-server`

### "Faial executable not found"

Check that Faial is installed in the container:
```bash
docker exec faial-mcp-server which faial-drf
docker exec faial-mcp-server faial-drf --version
```

### Slow analysis

- Increase timeout: `-e FAIAL_MCP_TIMEOUT_MS=300000` (5 minutes)
- Check container resources: `docker stats faial-mcp-server`
- Consider allocating more memory to Docker

### Permission errors

On Linux, if you get permission errors with mounted volumes:
```bash
docker run -p 8000:8000 \
  -v "$(pwd)/workspace:/workspace" \
  --user $(id -u):$(id -g) \
  faial-mcp
```

## Updating

To update to the latest Faial version:

```bash
# Rebuild the image (pulls latest Faial binaries)
docker-compose build --no-cache

# Restart the service
docker-compose up -d
```

## Stopping the Server

### Docker Run
Press `Ctrl+C` or `docker stop <container-id>`

### Docker Compose
```bash
docker-compose down
```

## Next Steps

- Read the [main README](README.md) for detailed MCP tool documentation
- Check [Faial documentation](https://gitlab.com/umb-svl/faial) for analysis options
- See `Faial-Optimal-Usage-Findings.md` for best practices on preparing kernels for analysis

## Support

- Faial issues: https://gitlab.com/umb-svl/faial/-/issues
- MCP issues: Open an issue in this repository

