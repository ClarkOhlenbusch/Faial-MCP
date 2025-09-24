# Claude Desktop Setup Guide for Faial MCP Server

This guide will walk you through setting up the Faial MCP server to work with Claude Desktop on Windows.

## ⚡ TL;DR - Quick Fix for Your Issue

**Your Railway approach failed because SSE doesn't work with Claude Desktop's MCP protocol.**

**Solution - Use Local Server:**
1. Edit: `%APPDATA%\Claude\claude_desktop_config.json`
2. Add this config:
```json
{
  "mcpServers": {
    "faial": {
      "command": "npm",
      "args": ["run", "mcp"],
      "cwd": "C:\\Users\\clark\\OneDrive\\Desktop\\faial-MCP"
    }
  }
}
```
3. Run: `npm install --save-dev cross-env` (if needed)
4. Test: `npm run mcp` (should show "stdio mode")
5. Restart Claude Desktop

**Why this works:** Uses proper stdio-based MCP communication instead of problematic HTTP/SSE.

## 🎯 Quick Setup (Recommended - Local Server)

### Step 1: Find Your Claude Desktop Config File

Claude Desktop stores its configuration in a JSON file. The location depends on your installation:

**Windows:**
- `%APPDATA%\Claude\claude_desktop_config.json`
- Or: `C:\Users\[YourUsername]\AppData\Roaming\Claude\claude_desktop_config.json`

**To find it easily:**
1. Press `Win + R`
2. Type: `%APPDATA%\Claude`
3. Press Enter
4. Look for `claude_desktop_config.json`

### Step 2: Set Up Local Server (Easiest)

**IMPORTANT:** The Railway hosted approach has issues with MCP protocol. Use the local server instead:

```json
{
  "mcpServers": {
    "faial": {
      "command": "npm",
      "args": ["run", "mcp"],
      "cwd": "C:\\Users\\clark\\OneDrive\\Desktop\\faial-MCP"
    }
  }
}
```

**Why Local Server is Better:**
- ✅ **Proper MCP protocol** - Direct stdio communication
- ✅ **No network issues** - No CORS or connection problems
- ✅ **Easier debugging** - Logs appear directly
- ✅ **Faster startup** - No HTTP connection overhead

### Step 3: Prepare Local Environment

Before using the local server:

1. **Navigate to project:**
   ```cmd
   cd C:\Users\clark\OneDrive\Desktop\faial-MCP
   ```

2. **Install dependencies:**
   ```cmd
   npm install
   ```

3. **Build the project:**
   ```cmd
   npm run build
   ```

### Step 3: Test the Setup

1. **Save the configuration file**
2. **Restart Claude Desktop completely**
   - Close Claude Desktop
   - Wait a few seconds
   - Reopen Claude Desktop
3. **Test the connection**
   - Open a new conversation in Claude Desktop
   - Ask Claude: "What MCP servers do you have access to?"
   - You should see "faial" listed

## 🔧 Local Development Setup (Detailed)

If you want to run the server locally for development:

### Prerequisites

1. **Node.js installed** (version 18 or higher)
2. **The faial-MCP project** on your machine

### Step-by-Step Local Setup

1. **Navigate to the project directory:**
   ```cmd
   cd C:\Users\clark\OneDrive\Desktop\faial-MCP
   ```

2. **Install dependencies:**
   ```cmd
   npm install
   ```

3. **Build the project:**
   ```cmd
   npm run build
   ```

4. **Test locally:**
   ```cmd
   npm run dev
   ```

   You should see: `Faial MCP server running in stdio mode`

5. **Configure Claude Desktop:**

   Edit `%APPDATA%\Claude\claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "faial": {
         "command": "npm",
         "args": ["run", "dev"],
         "cwd": "C:\\Users\\clark\\OneDrive\\Desktop\\faial-MCP"
       }
     }
   }
   ```

6. **Restart Claude Desktop**

## 🚀 Railway Hosted Setup (Advanced - Currently Not Working)

**⚠️ KNOWN ISSUE:** The Railway hosted approach currently has issues with the MCP protocol. The SSE endpoint hangs because it expects proper MCP initialization, not raw HTTP connections.

### Why Railway Hosting is Challenging:

1. **MCP Protocol Complexity**: MCP requires bidirectional communication with proper message framing
2. **SSE Limitations**: Server-sent events alone aren't sufficient for full MCP functionality
3. **Network Overhead**: HTTP-based MCP has additional complexity vs. stdio

### Railway Status:

✅ **Server Deployment**: Railway deployment works fine
✅ **Health Endpoint**: `https://faial-mcp-production.up.railway.app/health` returns success
✅ **Direct API**: REST endpoints work for direct file analysis
❌ **MCP Protocol**: SSE endpoint doesn't work with Claude Desktop

### Alternative: Use Railway for Direct API Calls

While the MCP protocol doesn't work, you can still use the Railway deployment for direct API calls:

```bash
# Test Faial analysis via Railway API
curl -X POST https://faial-mcp-production.up.railway.app/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"code": "int main() { return 0; }", "filename": "test.cu"}'
```

### Future: Possible Railway Solutions

To make Railway work with MCP, we would need:

1. **WebSocket Transport**: Replace SSE with WebSocket for bidirectional communication
2. **MCP WebSocket Client**: Create a proper WebSocket-based MCP client
3. **Protocol Bridge**: Bridge between Claude Desktop's stdio and WebSocket

For now, **use the local server setup** which works perfectly with Claude Desktop.

## 🧪 Testing Your Setup

### Test 1: Configuration Validation

1. **Open Claude Desktop**
2. **Start a new conversation**
3. **Ask:** "What tools do you have access to?"
4. **Expected response:** Claude should mention Faial analysis tools

### Test 2: Faial Functionality

1. **Ask Claude:** "Can you analyze this CUDA code for me?"
2. **Provide sample code:**
   ```cuda
   __global__ void hello() {
       printf("Hello from GPU!\n");
   }

   int main() {
       hello<<<1,1>>>();
       cudaDeviceSynchronize();
       return 0;
   }
   ```
3. **Expected:** Claude should use the Faial tools to analyze the code

## 🔍 Troubleshooting

### Problem: "No MCP servers found"

**Solution:**
1. Check the config file location: `%APPDATA%\Claude\claude_desktop_config.json`
2. Verify JSON syntax (use a JSON validator)
3. Make sure file paths use double backslashes: `\\`
4. Restart Claude Desktop completely

### Problem: "Connection failed" or Railway SSE hanging

**This is the error you encountered with Railway setup.**

**Root Cause:** The Railway SSE approach doesn't work because:
- Claude Desktop expects stdio-based MCP communication
- SSE endpoint waits for proper MCP initialization
- Raw HTTP connections don't follow MCP protocol

**Solution:** Use local server instead:

```json
{
  "mcpServers": {
    "faial": {
      "command": "npm",
      "args": ["run", "mcp"],
      "cwd": "C:\\Users\\clark\\OneDrive\\Desktop\\faial-MCP"
    }
  }
}
```

**For Local Setup:**
1. Ensure the project is built: `npm run build`
2. Test MCP mode works: `npm run mcp` (should show "Faial MCP server running in stdio mode")
3. Verify the `cwd` path in config is correct
4. Make sure npm and node are in your PATH
5. Install cross-env if missing: `npm install --save-dev cross-env`

### Problem: "Faial not working"

**Expected behavior:** The system will show helpful error messages about Faial installation
1. Check Railway logs for Faial build status
2. The server includes fallback error messages
3. Test with the `/health` endpoint first

### Problem: "JSON syntax error"

**Common mistakes:**
```json
❌ Wrong:
{
  "mcpServers": {
    "faial": {
      "command": "npm",
      "args": ["run", "dev"],
      "cwd": "C:\Users\clark\..." // Single backslash
    }
  }
}

✅ Correct:
{
  "mcpServers": {
    "faial": {
      "command": "npm",
      "args": ["run", "dev"],
      "cwd": "C:\\Users\\clark\\..." // Double backslash
    }
  }
}
```

## 📂 File Locations Reference

- **Claude Desktop Config:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Project Directory:** `C:\Users\clark\OneDrive\Desktop\faial-MCP`
- **Client Script:** `C:\Users\clark\OneDrive\Desktop\faial-MCP\faial-client.js`
- **Railway Health Check:** `https://faial-mcp-production.up.railway.app/health`

## ✅ Success Checklist

- [ ] Claude Desktop config file exists and has valid JSON
- [ ] File paths use double backslashes (`\\`)
- [ ] Claude Desktop has been restarted
- [ ] Health endpoint returns success (for Railway setup)
- [ ] Claude recognizes the Faial MCP server
- [ ] Claude can use Faial analysis tools

## 💡 Pro Tips

1. **Start with local setup first** - it's easier to debug
2. **Use Railway for production** - no need to keep your computer running
3. **Check Railway logs** - they show detailed build and runtime information
4. **Test health endpoint first** - ensures basic connectivity
5. **Use JSON validator** - prevents configuration syntax errors

## 🆘 Still Having Issues?

If you're still having problems:

1. **Check the exact error message** in Claude Desktop
2. **Verify Railway deployment** at the health endpoint
3. **Test locally first** with `npm run dev`
4. **Check file paths** are correct and accessible
5. **Restart Claude Desktop** after any config changes

The most common issue is file path formatting on Windows - make sure to use double backslashes!