# Faial MCP Server - Configuration Examples

This document provides configuration examples for connecting to the hosted Faial MCP server from various MCP clients.

## 🚀 Quick Start

Once the server is deployed (recommended: Railway), users just need to add this simple configuration to their MCP client:

**Important**: The hosted server includes Faial CLI built-in - no local installation required!

### Railway Deployment (Recommended)

Railway automatically builds and deploys the Docker container with Faial CLI pre-installed.

**Why Railway?**
- ✅ **Container-native**: Perfect for running Faial CLI
- ✅ **No serverless limitations**: Full file system access
- ✅ **Easy debugging**: Full container logs
- ✅ **Automatic deployments**: Connect GitHub repo for auto-deploy

**Railway Setup:**
1. Sign up at [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Railway automatically builds and deploys the container
4. Get your app URL from Railway dashboard (e.g., `https://your-app-name.railway.app`)

**Recommended Configuration:**
```json
{
  "mcpServers": {
    "faial": {
      "command": "node",
      "args": ["faial-client.js"]
    }
  }
}
```

**Setup:**
1. Download the `faial-client.js` file from your repository
2. Place it in your MCP configuration directory
3. Use the configuration above

**What this does:**
- Uses Node.js to run the client script
- Connects to your Railway-hosted MCP server
- Handles the HTTP-based MCP protocol correctly
- No dependency on browser `fetch` API

**Testing Your Deployment:**
1. Visit: `https://your-railway-app.railway.app/health`
2. Should return: `{"status":"ok","service":"faial-mcp","version":"0.1.0"}`
3. Check Railway logs if there are issues

## MCP Client Configurations

### Cursor

Add to your Cursor settings or `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "faial": {
      "command": "fetch",
      "args": ["https://your-domain.com/sse"]
    }
  }
}
```

### Claude Code

Add to your MCP configuration file:

```json
{
  "mcpServers": {
    "faial": {
      "command": "fetch",
      "args": ["https://your-domain.com/sse"]
    }
  }
}
```

### VS Code with MCP Extension

In your VS Code settings:

```json
{
  "mcp.servers": {
    "faial": {
      "command": "fetch",
      "args": ["https://your-domain.com/sse"]
    }
  }
}
```

### MCP Inspector

```bash
# Test the hosted server
mcp-inspector fetch https://your-domain.com/sse
```

## Alternative: Direct API Usage

If your MCP client doesn't support HTTP-based servers, users can use the REST API directly:

### File Upload Analysis

```bash
curl -X POST https://your-domain.com/analyze/upload \
  -F "file=@your_file.cu" \
  -F "extraArgs=[]" \
  -F "timeoutMs=60000"
```

### Text Analysis

```bash
curl -X POST https://your-domain.com/analyze/text \
  -H "Content-Type: application/json" \
  -d '{
    "code": "your cuda code here",
    "filename": "main.cu",
    "extraArgs": [],
    "timeoutMs": 60000
  }'
```

### Health Check

```bash
curl https://your-domain.com/health
```

## Environment Variables

The hosted server respects these environment variables:

- `TRANSPORT_MODE`: Set to `http` for hosted mode (default)
- `PORT`: Port number (default: 3000)
- `FAIAL_PATH`: Path to Faial CLI executable

## Troubleshooting

1. **Connection Issues**: Ensure the server URL is accessible and the `/sse` endpoint is reachable
2. **Analysis Failures**: Check the health endpoint first: `https://your-domain.com/health`
3. **Timeouts**: Increase timeout values for large codebases
4. **CORS Issues**: The server includes proper CORS headers for cross-origin requests

## Security Notes

- The server includes CORS support for web-based MCP clients
- File uploads are processed in temporary directories and cleaned up automatically
- Analysis is performed in isolated temporary files
- No persistent storage of uploaded files
