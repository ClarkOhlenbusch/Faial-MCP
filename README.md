# Faial MCP Server

Model Context Protocol (MCP) server that wraps the Faial CUDA static analysis CLI.

This lets AI agents (Claude Code, MCP Inspector, etc.) analyze CUDA/C++ codebases via Faial using standard MCP tools.

## 🚀 Quick Start (Hosted Version)

**For Users**: Just add this configuration to your MCP client:

```json
{
  "mcpServers": {
    "faial": {
      "command": "fetch",
      "args": ["https://your-hosted-server.com/sse"]
    }
  }
}
```

**For Developers**: Deploy your own hosted instance in minutes using the provided configurations.

## Deployment Options

### 🚀 Option 1: Vercel (Recommended)

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Deploy**:
   ```bash
   # Deploy to Vercel
   ./deploy.sh

   # Or manually:
   npm run build
   vercel --prod
   ```

3. **Update Faial CLI**: Edit the Dockerfile to install Faial CLI, then redeploy.

### 🐳 Option 2: Docker

1. **Build and run locally**:
   ```bash
   docker-compose up --build
   ```

2. **Deploy to cloud**:
   - Push to Docker registry
   - Deploy container to any cloud platform (AWS, GCP, Azure, etc.)

### 🌐 Option 3: Other Platforms

Use the provided configuration files:
- `vercel.json` for Vercel
- `netlify.toml` for Netlify
- `Dockerfile` for container platforms

### 🔧 Manual Setup

If you prefer manual deployment:

1. **Build the application**:
   ```bash
   npm install
   npm run build
   ```

2. **Set environment variables**:
   ```bash
   export TRANSPORT_MODE=http
   export PORT=3000
   export FAIAL_PATH=/path/to/faial
   ```

3. **Deploy to your platform** (Heroku, Railway, Render, etc.)

## Prerequisites

- Node.js 18+
- Faial CLI installed and on your `PATH`, or set `FAIAL_PATH` to the executable.
- Windows PowerShell is supported; commands below use PowerShell syntax.

## Install

```powershell
# From this folder
npm install
npm run build
```

Optionally set the executable path if Faial isn't on `PATH`:

```powershell
$env:FAIAL_PATH = "C:\\path\\to\\faial.exe"
```

### Windows + WSL (Recommended for Faial)

Faial currently targets Linux/macOS. On Windows, run it via WSL:

1) Install WSL and a Linux distro (e.g., Ubuntu), then install Faial inside WSL.
2) In this server, set the following env vars so we invoke Faial through WSL:

```powershell
# Force WSL mode
$env:FAIAL_MODE = "wsl"

# Optional: choose a specific distro name (output of `wsl -l -v`)
$env:FAIAL_WSL_DISTRO = "Ubuntu"

# Path to Faial inside WSL (if not on PATH there)
$env:FAIAL_WSL_PATH = "/usr/local/bin/faial"
```

We automatically translate Windows paths to WSL paths (e.g., `C:\Users\you\file.cu` -> `/mnt/c/Users/you/file.cu`) and set the working directory inside WSL if provided.

## Run

```powershell
# Start the MCP server over stdio
npm start
# or directly
node dist/index.js
```

Use with MCP clients such as:
- MCP Inspector: https://github.com/modelcontextprotocol/inspector
- Claude Code: add a custom server entry using the `faial-mcp` command.

## Hosted Server Features

When deployed, the server provides:

### MCP Protocol Endpoints
- `GET /sse` - Server-Sent Events endpoint for MCP protocol
- `POST /messages` - Message handling for MCP protocol
- `GET /health` - Health check endpoint

### REST API Endpoints
- `POST /analyze/upload` - Upload and analyze CUDA files
- `POST /analyze/text` - Analyze raw CUDA code

### Configuration
- **Environment**: Set `TRANSPORT_MODE=http` (default when hosted)
- **Port**: Configure via `PORT` environment variable (default: 3000)
- **CORS**: Enabled for cross-origin requests

## Tools

- `faial.get_version`: Verify Faial installation; returns the CLI version.
- `faial.analyze_file`: Analyze a single file.
  - Args: `path` (string), `cwd?` (string), `extraArgs?` (string[]), `timeoutMs?` (number)
- `faial.analyze_directory`: Analyze an entire directory.
  - Args: `dir` (string), `pattern?` (string), `recursive?` (boolean), `extraArgs?` (string[]), `timeoutMs?` (number)
- `faial.analyze_text`: Analyze raw source text by writing it to a temp file.
  - Args: `code` (string), `filename?` (string), `extraArgs?` (string[]), `timeoutMs?` (number)

Outputs prefer JSON (when Faial supports `--format json`), falling back to plain text.

## User Configuration

See [MCP_CONFIG_EXAMPLES.md](MCP_CONFIG_EXAMPLES.md) for detailed configuration examples for various MCP clients.

## Configuration

- `FAIAL_PATH`: Absolute path to the Faial executable. If unset, the server searches the system `PATH`.
- Timeouts: Each tool accepts `timeoutMs` to prevent hanging analyses.

## Notes and Assumptions

- The commands assume Faial supports subcommand `analyze` and `--format json`. If your Faial CLI differs, adjust argument building in `src/index.ts` accordingly.
- Windows is supported; process execution uses `spawn` and `windowsHide`.

## Development

```powershell
npm run dev   # run from sources using ts-node
npm run check # typecheck
npm run build # compile to dist
```

## License

MIT