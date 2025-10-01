# MCP Configuration Guide for Faial

There are **two ways** to use Faial MCP Server, depending on your client:

## Option 1: stdio Transport (Recommended for Claude Desktop)

This spawns a new container for each MCP session. Best for local development.

### Claude Desktop Configuration

**Location:**
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

**Configuration:**

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

**With workspace access (to analyze local files):**

```json
{
  "mcpServers": {
    "faial": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-v",
        "C:\\Users\\clark\\OneDrive\\Desktop\\Faial-MCP\\workspace:/workspace",
        "faial-mcp",
        "faial-mcp-server",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

**Note:** Adjust the path `C:\\Users\\clark\\OneDrive\\Desktop\\Faial-MCP\\workspace` to your actual workspace directory.

### Testing stdio Configuration

After adding the config:

1. **Restart Claude Desktop** completely (quit and reopen)
2. Start a new conversation
3. Look for the 🔨 tools icon - you should see `analyze_kernel` available
4. Try asking: *"What tools do you have available?"*
5. Claude should mention the `analyze_kernel` tool for Faial analysis

---

## Option 2: SSE Transport (Network Server - Advanced)

This connects to the **running server** at `http://localhost:8000`. Good for:
- Multiple clients sharing one server
- Remote access
- Persistent server deployment

### Important: MCP Client Support

**⚠️ Note:** As of now, Claude Desktop's primary transport is `stdio`. SSE support may be limited or require additional configuration.

For SSE transport, you typically need:
- Custom MCP clients
- Browser-based clients
- Or MCP-compatible applications with SSE support

### Cursor Configuration (if it supports SSE)

In Cursor settings, add MCP server:

```json
{
  "mcp": {
    "servers": {
      "faial": {
        "url": "http://localhost:8000",
        "transport": "sse"
      }
    }
  }
}
```

### Testing SSE Server

Your server is already running! Test it with:

```powershell
# Check if the server is responding
curl http://localhost:8000

# View server logs
docker-compose logs -f
```

---

## Which Option Should You Use?

### Use stdio (Option 1) if:
- ✅ You're using **Claude Desktop**
- ✅ You want the **simplest setup**
- ✅ You're doing **local development**
- ✅ One user, one machine

### Use SSE (Option 2) if:
- ✅ You need **multiple clients** to share one server
- ✅ You're deploying to a **remote server**
- ✅ Your MCP client **supports SSE transport**
- ✅ You want a **persistent server**

---

## Complete Example: Claude Desktop Setup

### Step 1: Stop the SSE Server (Optional)

If you're using stdio, you don't need the running server:

```powershell
docker-compose down
```

### Step 2: Create/Edit Claude Config

Open: `%APPDATA%\Claude\claude_desktop_config.json`

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

### Step 3: Restart Claude Desktop

Completely quit and reopen Claude Desktop.

### Step 4: Test It!

In a new Claude conversation, ask:

```
What MCP tools do you have available?
```

Claude should respond with something like:
> I have access to the `analyze_kernel` tool from Faial, which can analyze CUDA and WGSL kernels for data races...

### Step 5: Analyze a Kernel

Try this:

```
Analyze this CUDA kernel for data races:

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

Claude will use the `analyze_kernel` tool and return Faial's analysis!

---

## Troubleshooting

### Claude Doesn't See the Tool

1. **Verify config file location**:
   ```powershell
   notepad %APPDATA%\Claude\claude_desktop_config.json
   ```

2. **Check JSON syntax** - make sure it's valid JSON (no trailing commas!)

3. **Restart Claude Desktop** - fully quit and reopen

4. **Check Docker** - ensure Docker Desktop is running:
   ```powershell
   docker ps
   ```

5. **Test the command manually**:
   ```powershell
   docker run -i --rm faial-mcp faial-mcp-server --transport stdio
   ```
   Type `Ctrl+C` to exit.

### Tool Calls Fail

1. **Check Docker permissions** - ensure Docker can run containers

2. **View Claude logs** (Developer Console):
   - Windows: `%APPDATA%\Claude\logs\`
   - Look for MCP-related errors

3. **Test with a simple kernel** - start with basic code before complex examples

### "Container not found" Error

Rebuild the image:
```powershell
docker build -t faial-mcp .
```

---

## Advanced: Multiple MCP Servers

You can have Faial alongside other MCP servers:

```json
{
  "mcpServers": {
    "faial": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "faial-mcp",
        "faial-mcp-server",
        "--transport", "stdio"
      ]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/workspace"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-api-key"
      }
    }
  }
}
```

---

## Next Steps

1. ✅ Configure Claude Desktop with the stdio config above
2. ✅ Restart Claude Desktop
3. ✅ Test with a simple CUDA kernel
4. ✅ Try the example in `workspace/example.cu`

For SSE transport and remote hosting, see [DOCKER.md](DOCKER.md).

