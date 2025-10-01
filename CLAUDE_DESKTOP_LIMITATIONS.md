# Claude Desktop MCP Limitations

## ⚠️ Important: Claude Desktop Does NOT Support SSE Transport

If you've deployed Faial MCP to a cloud platform (Railway, Render, AWS, etc.) and are trying to connect Claude Desktop to it via SSE, **it won't work**.

### What Claude Desktop Supports

Claude Desktop **ONLY** supports:
- ✅ `stdio` transport (standard input/output)
- ✅ Local command execution (`command` + `args` in config)
- ✅ Processes that run on your local machine

### What Claude Desktop Does NOT Support

- ❌ Remote HTTP/HTTPS URLs
- ❌ SSE (Server-Sent Events) transport
- ❌ WebSocket connections
- ❌ Network-based MCP servers

---

## Why This Configuration Won't Work

```json
{
  "mcpServers": {
    "faial": {
      "url": "https://faial-mcp-production-b547.up.railway.app",
      "transport": "sse"
    }
  }
}
```

**Reason:** Claude Desktop expects a `command` field, not a `url` field.

---

## ✅ Correct Configuration for Claude Desktop

### Option 1: Local Docker (Recommended)

```json
{
  "mcpServers": {
    "faial": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "faial-mcp",
        "faial-mcp-server",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

### Option 2: Pull from Registry

If you've published your image to Docker Hub or another registry:

```json
{
  "mcpServers": {
    "faial": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "registry.railway.app/your-image:latest",
        "faial-mcp-server",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

Users would first run:
```bash
docker pull registry.railway.app/your-image:latest
```

---

## When Is Railway/Cloud Deployment Useful?

Your Railway deployment **IS** useful for:

### 1. API Integrations
Direct HTTP API calls to your MCP server:
```bash
curl -X POST https://faial-mcp-production-b547.up.railway.app/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "analyze_kernel", "arguments": {...}}'
```

### 2. Custom MCP Clients
If you build a custom MCP client that supports SSE:
```python
import httpx

client = httpx.Client()
response = client.get("https://faial-mcp-production-b547.up.railway.app/sse")
```

### 3. Web Applications
Browser-based apps that can consume SSE streams.

### 4. Future MCP Clients
As the MCP ecosystem grows, more clients may support network transports.

---

## Recommended Architecture

### For Individual Users (Claude Desktop)

```
┌─────────────┐
│   Claude    │
│  Desktop    │
└──────┬──────┘
       │ stdio
       ▼
┌─────────────┐
│   Docker    │
│  Container  │  (runs locally on user's machine)
│  faial-mcp  │
└─────────────┘
```

**Config:**
```json
{
  "mcpServers": {
    "faial": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "faial-mcp", "faial-mcp-server", "--transport", "stdio"]
    }
  }
}
```

### For Team/API Access (Cloud Deployment)

```
┌─────────────┐
│   Custom    │
│   Client    │
└──────┬──────┘
       │ HTTPS/SSE
       ▼
┌─────────────┐
│   Railway   │
│  faial-mcp  │  (hosted remotely)
│   Server    │
└─────────────┘
```

**Access:**
```bash
curl https://faial-mcp-production-b547.up.railway.app
```

---

## How to Distribute Your MCP Server

### Method 1: Docker Hub (Public)

1. **Build and tag:**
   ```bash
   docker build -t yourusername/faial-mcp:latest .
   ```

2. **Push to Docker Hub:**
   ```bash
   docker push yourusername/faial-mcp:latest
   ```

3. **Users pull and configure:**
   ```bash
   docker pull yourusername/faial-mcp:latest
   ```
   
   ```json
   {
     "mcpServers": {
       "faial": {
         "command": "docker",
         "args": ["run", "-i", "--rm", "yourusername/faial-mcp", "faial-mcp-server", "--transport", "stdio"]
       }
     }
   }
   ```

### Method 2: GitHub Container Registry

1. **Build and tag:**
   ```bash
   docker build -t ghcr.io/yourusername/faial-mcp:latest .
   ```

2. **Push to GHCR:**
   ```bash
   docker push ghcr.io/yourusername/faial-mcp:latest
   ```

3. **Users configure:**
   ```json
   {
     "mcpServers": {
       "faial": {
         "command": "docker",
         "args": ["run", "-i", "--rm", "ghcr.io/yourusername/faial-mcp", "faial-mcp-server", "--transport", "stdio"]
       }
     }
   }
   ```

### Method 3: Railway + stdio Wrapper (Advanced)

You could create a small Node.js/Python wrapper that:
1. Connects to your Railway SSE endpoint
2. Translates between stdio and SSE
3. Runs as a local command for Claude Desktop

This is complex but allows cloud hosting + Claude Desktop.

---

## Summary

| Use Case | Solution | Transport |
|----------|----------|-----------|
| Claude Desktop | Local Docker container | stdio |
| API Integration | Railway deployment | SSE/HTTP |
| Custom Client | Railway deployment | SSE/HTTP |
| Team Access | Railway deployment | SSE/HTTP |
| Distribution | Docker registry + stdio | stdio |

**Bottom Line:**
- ✅ Railway deployment works great for APIs and custom clients
- ❌ Railway deployment **won't work** with Claude Desktop directly
- ✅ Use local Docker + stdio for Claude Desktop
- ✅ Consider publishing to Docker Hub for easy distribution

