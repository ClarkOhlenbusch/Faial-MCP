# Railway Deployment Guide for Faial MCP

## Current Issues Fixed

1. **Faial CLI Installation**: Enhanced Dockerfile with better build detection and error handling
2. **SSE Endpoint**: Fixed server-sent events connection with proper headers
3. **Error Messages**: Improved error reporting when Faial binary is missing

## Deployment Steps

1. **Build and Deploy**:
   ```bash
   # Railway will automatically detect Dockerfile and build
   railway up
   ```

2. **Verify Deployment**:
   ```bash
   # Test health endpoint
   curl https://your-app.up.railway.app/health

   # Test Faial version (will show if binary is installed)
   curl -X POST -H "Content-Type: application/json" \
     -d '{}' https://your-app.up.railway.app/analyze/text
   ```

## MCP Configuration for Claude Desktop

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "faial": {
      "command": "node",
      "args": ["-e", "
        const { SSETransport } = require('@modelcontextprotocol/sdk/client/sse.js');
        const transport = new SSETransport('https://faial-mcp-production.up.railway.app/sse');
        transport.start();
      "]
    }
  }
}
```

## Alternative: Direct HTTP Client

For testing, use the HTTP endpoints:

- Health: `GET /health`
- File Upload: `POST /analyze/upload` (with multipart file)
- Text Analysis: `POST /analyze/text` (with JSON body: `{"code": "your code"}`)

## Troubleshooting

1. **Faial Not Found**: The Docker build will create a stub binary if Faial can't be built properly. Check deployment logs for build details.

2. **SSE Connection Issues**: The SSE endpoint now has proper headers and error handling. If still having issues, check network/firewall settings.

3. **CORS Issues**: Server allows Claude.ai, Cursor.sh, and MCP origins. Add your domain if needed.

## Next Steps

If Faial repository doesn't have clear build instructions, consider:
1. Contacting the Faial project maintainers
2. Using an alternative CUDA static analysis tool
3. Creating a mock implementation for testing MCP functionality