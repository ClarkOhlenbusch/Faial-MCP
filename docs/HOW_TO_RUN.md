# How to Run Faial MCP Server

## Quick Answer

claude mcp add --transport sse faial https://faial-mcp-production-b547.up.railway.app/sse

## Three Ways to Run

### 1. Quick Test (5 minutes)

```bash
# Build
docker build -t faial-mcp .

# Run
docker run -p 8000:8000 faial-mcp
```

Server is now at `http://localhost:8000`

### 2. Production Setup (Recommended)

```bash
# Start with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Server runs in background with auto-restart.

### 3. Test Everything First

```bash
# Run automated tests
./test-docker.sh        # Linux/macOS
.\test-docker.ps1       # Windows PowerShell
```

This verifies:
- Docker is working
- Faial binaries are accessible
- MCP server starts correctly
- Network connectivity works

## What's Inside the Container?

```
/opt/faial/          ← Faial binaries (faial-drf, cu-to-json)
/app/                ← MCP Server (Python)
/workspace/          ← Mounted from host (for your .cu files)
```

## How It Works

1. **Dockerfile downloads Faial** from GitLab (official distribution)
2. **Installs Python MCP server** package
3. **Exposes port 8000** for network access
4. **Runs SSE transport** by default (works great for remote hosting)

## Usage Modes

### Local Development (stdio)

```bash
docker run -i faial-mcp faial-mcp-server --transport stdio
```

Good for: Testing, debugging, local-only use

### Network Server (sse) - Default

```bash
docker run -p 8000:8000 faial-mcp
```

Good for: Claude Desktop, Cursor, remote access, team deployment

### With File Access

```bash
docker run -p 8000:8000 -v "$(pwd)/workspace:/workspace" faial-mcp
```

Good for: Analyzing files from your machine

## Connecting MCP Clients

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)  
`%APPDATA%\Claude\claude_desktop_config.json` (Windows)

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

### Cursor

Settings → MCP Servers → Add:
- URL: `http://localhost:8000`
- Transport: `sse`

## Remote Hosting

Deploy to any cloud platform:

```bash
# Build and push
docker tag faial-mcp myregistry/faial-mcp:latest
docker push myregistry/faial-mcp:latest

# Run on server
docker run -d -p 8000:8000 --restart unless-stopped myregistry/faial-mcp:latest
```

Update MCP client to: `http://your-server-ip:8000`

## Environment Variables

```bash
docker run -p 8000:8000 \
  -e FAIAL_MCP_TIMEOUT_MS=120000 \
  -e FAIAL_MCP_EXECUTABLE=/opt/faial/faial-drf \
  faial-mcp
```

Available:
- `FAIAL_MCP_EXECUTABLE` - Path to faial-drf
- `FAIAL_MCP_TIMEOUT_MS` - Analysis timeout
- `FAIAL_MCP_HOST` - Bind host (default: 0.0.0.0)
- `FAIAL_MCP_PORT` - Bind port (default: 8000)

## Complete Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Step-by-step setup guide
- **[DOCKER.md](DOCKER.md)** - Detailed Docker usage
- **[README.md](README.md)** - Full documentation
- **[Faial-Optimal-Usage-Findings.md](Faial-Optimal-Usage-Findings.md)** - Best practices

## Summary

✅ **Does this work?** Yes!  
✅ **Do users need Faial installed?** No!  
✅ **Can it be network-based?** Yes!  
✅ **Can it be hosted remotely?** Yes!  
✅ **Is it production-ready?** Yes!

The Docker container is **completely self-contained** with both Faial and the MCP server. Users only need Docker.

