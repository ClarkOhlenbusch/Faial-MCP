# Getting Started with Faial MCP Server

This guide will help you set up and run the Faial MCP Server using Docker.

## What is This?

**Faial MCP Server** is a dockerized service that:
- ✅ Runs Faial (CUDA static analysis tool) without local installation
- ✅ Exposes Faial through the Model Context Protocol (MCP)
- ✅ Lets AI assistants like Claude analyze CUDA kernels for data races
- ✅ Can be hosted remotely for team-wide access

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- An MCP-compatible client (Claude Desktop, Cursor, etc.)

## Step-by-Step Setup

### 1. Build the Docker Image

Open a terminal in this directory and run:

```bash
docker build -t faial-mcp .
```

This will:
- Download Ubuntu 24.04 base image
- Install Python 3 and dependencies
- Download Faial binaries (latest version from GitLab)
- Set up the MCP server

**Build time:** ~2-5 minutes (depending on your connection)  
**Image size:** ~500MB

### 2. Start the Server

**Option A: Docker Run (Quick Test)**
```bash
docker run -p 8000:8000 faial-mcp
```

**Option B: Docker Compose (Recommended)**
```bash
docker-compose up -d
```

You should see output like:
```
Starting faial-mcp-server ... done
```

### 3. Verify It's Running

Test the server with curl:

```bash
curl http://localhost:8000/health
```

Or check Docker:
```bash
docker ps
```

You should see `faial-mcp-server` running.

### 4. Configure Your MCP Client

#### For Claude Desktop:

1. Find your config file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. Add this configuration:
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

3. Restart Claude Desktop

#### For Cursor:

1. Open Cursor Settings → MCP Servers
2. Add new server:
   - Name: `faial`
   - URL: `http://localhost:8000`
   - Transport: `sse`
3. Save and restart Cursor

### 5. Test the Integration

Ask your AI assistant:

```
Analyze this CUDA kernel for data races:

__global__ void add(int *a, int *b, int *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

The AI should use the `analyze_kernel` tool and return Faial's analysis results!

## Using the Server

### Analyzing Inline Code

Just paste CUDA code in your conversation. The AI will send it to Faial for analysis.

### Analyzing Files

1. Place your `.cu` files in the `workspace/` directory
2. Reference them in your conversation:

```
Analyze the kernel in /workspace/my_kernel.cu
```

The server automatically mounts the `workspace/` directory, so files there are accessible.

### Advanced Options

You can pass Faial-specific options through the AI:

```
Analyze this kernel with grid-level analysis:

[your CUDA code]

Use --grid-level flag
```

The AI will pass the appropriate parameters to Faial.

## Common Tasks

### Stop the Server

**Docker Run**: Press `Ctrl+C`

**Docker Compose**:
```bash
docker-compose down
```

### View Logs

```bash
# Docker Compose
docker-compose logs -f

# Docker Run
docker logs <container-id>
```

### Update Faial

To get the latest Faial version:

```bash
docker-compose build --no-cache
docker-compose up -d
```

### Change Port

Edit `docker-compose.yml` and change `8000:8000` to `<host-port>:8000`, then:

```bash
docker-compose up -d
```

Update your MCP client config to use the new port.

### Deploy to Remote Server

1. Build and push to a registry:
```bash
docker tag faial-mcp your-registry/faial-mcp:latest
docker push your-registry/faial-mcp:latest
```

2. On the remote server:
```bash
docker pull your-registry/faial-mcp:latest
docker run -d -p 8000:8000 --name faial-mcp your-registry/faial-mcp:latest
```

3. Update your MCP client to use `http://your-server:8000`

## Architecture

```
┌─────────────────┐
│   MCP Client    │  (Claude Desktop, Cursor, etc.)
│  (Your Machine) │
└────────┬────────┘
         │ HTTP/SSE
         ▼
┌─────────────────┐
│  Faial MCP      │  (Docker Container)
│  Server         │
│  ┌───────────┐  │
│  │ FastMCP   │  │  (Python MCP Server)
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ faial-drf │  │  (Faial Analysis Engine)
│  └───────────┘  │
└─────────────────┘
```

## Troubleshooting

### "Connection refused"
- Ensure container is running: `docker ps`
- Check port mapping: `docker port faial-mcp-server`
- Verify firewall isn't blocking port 8000

### "Tool not available"
- Restart your MCP client after adding the config
- Check the config file syntax (valid JSON)
- Verify the server URL is correct

### "Analysis failed"
- Check container logs: `docker-compose logs -f`
- Ensure your CUDA code is well-formed
- Try increasing timeout: add `-e FAIAL_MCP_TIMEOUT_MS=120000` to docker run

### "Permission denied" on workspace files
On Linux, you may need to match user permissions:
```bash
docker run -p 8000:8000 \
  -v "$(pwd)/workspace:/workspace" \
  --user $(id -u):$(id -g) \
  faial-mcp
```

## Next Steps

- 📖 Read [DOCKER.md](DOCKER.md) for detailed Docker usage
- 📖 Read [README.md](README.md) for MCP tool documentation
- 📖 Read [Faial-Optimal-Usage-Findings.md](Faial-Optimal-Usage-Findings.md) for analysis best practices
- 🔧 Explore the example kernel in `workspace/example.cu`

## Need Help?

- **Faial Issues**: https://gitlab.com/umb-svl/faial/-/issues
- **MCP Issues**: Open an issue in this repository
- **Docker Issues**: Check [Docker documentation](https://docs.docker.com/)

## Summary

You now have a fully functional Faial MCP Server running in Docker! 🎉

Your AI assistant can now:
- ✅ Analyze CUDA kernels for data races
- ✅ Check for synchronization bugs
- ✅ Verify memory safety
- ✅ Find race conditions

All without installing Faial locally!

