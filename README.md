# Faial MCP Server

This package wraps the Faial CUDA static analysis binaries in a Model Context Protocol (MCP) server. It allows agents and other MCP clients to invoke `faial-drf` through the `analyze_kernel` tool.

## Quick Start with Docker (Recommended)

The easiest way to use Faial MCP is via Docker, which bundles both Faial binaries and the MCP server:

```bash
# Test your Docker setup (optional but recommended)
# Linux/macOS:
./test-docker.sh
# Windows PowerShell:
.\test-docker.ps1

# Build the Docker image
docker build -t faial-mcp .

# Run as a network-based MCP server (SSE transport)
docker run -p 8000:8000 faial-mcp

# Or run with stdio transport (for local development)
docker run -i faial-mcp faial-mcp-server --transport stdio
```

See the [Docker Usage](#docker-usage) section below for detailed instructions, or read [GETTING_STARTED.md](GETTING_STARTED.md) for a complete walkthrough.

## Prerequisites (Local Installation)

- Python 3.10 or newer
- The Faial binaries (`faial-drf`, `cu-to-json`, supporting libraries) available on the host system

## Local Installation

These instructions are recommended for all platforms, including WSL (Windows Subsystem for Linux).

1.  **Create a virtual environment:**

    ```bash
    python3 -m venv .venv
    ```

2.  **Activate the virtual environment:**

    -   **Windows (Command Prompt/PowerShell):**

        ```bash
        .venv\Scripts\activate
        ```

    -   **Linux/macOS/WSL:**

        ```bash
        source .venv/bin/activate
        ```

3.  **Install the package:**

    ```bash
    pip install -e .
    ```

    You may install the optional development dependencies with `pip install -e .[dev]`.

## Running

With the virtual environment activated, run the server:

```bash
faial-mcp-server --transport stdio
```

The server defaults to the stdio transport, which is suitable for local agent integrations. SSE and Streamable HTTP transports are also available via `--transport sse` or `--transport streamable-http`.

### Configuration

Environment variables configure default paths:

- `FAIAL_MCP_EXECUTABLE`: Path to `faial-drf` (defaults to looking up `faial-drf` on `PATH`).
- `FAIAL_MCP_CU_TO_JSON`: Path to `cu-to-json` helper when the default lookup is insufficient.
- `FAIAL_MCP_TIMEOUT_MS`: Default timeout in milliseconds applied when a request does not specify one.
- `FAIAL_MCP_TRANSPORT`: Default transport (`stdio`, `sse`, or `streamable-http`).
- `FAIAL_MCP_HOST` / `FAIAL_MCP_PORT`: Defaults for host/port when using SSE or HTTP transports.

Each tool invocation can override the executable, helper paths, working directory, and environment.

## Available Tools

### `analyze_kernel`

Runs Faial's data-race freedom analysis (`faial-drf`) against a CUDA or WGSL kernel. **All requests must provide kernel source inline via the `source` field.** File-path based submissions are not supported because the server writes each request to an isolated temporary file before calling `faial-drf`.

Key arguments:

- `source` (required): Self-contained kernel snippet as a string. Include any helper functions, structs, constants, and macros so the kernel can compile without additional files.
- `virtual_filename`: Logical filename (e.g., `my_kernel.cu`) to control the extension recorded in logs.
- `include_dirs`: List of directories to forward via `-I`. Useful when a kernel legitimately depends on
  headers you can mount into the server's workspace. Paths are resolved relative to `working_directory`
  (default: the server's current working directory), so provide absolute paths or set
  `working_directory` when referencing mounted files.
- `macros`: Dictionary of macro definitions (`{"DEBUG": "1", "USE_FAST": null}`) that becomes `-D` flags. Values are optional; use `null` for flag-style macros. CLI-style strings such as `"DEBUG 1"` or `"USE_FAST=value"` are also accepted and normalized automatically.
- `working_directory`: Directory in which to run `faial-drf`. This also controls how relative include
  paths and helper binaries are resolved. Defaults to the server's launch directory.
- `include_raw_output`: Set to `true` when you need the complete stdout/stderr/JSON payload from Faial.
  By default, the MCP response only includes concise summaries and short excerpts to keep payloads
  small.
- `params`, `block_dim`, `grid_dim`, `only_kernel`, `only_array`, `logic`, `timeout_ms`,
  `find_true_data_races`, `grid_level`, `all_levels`, `all_dims`, `unreachable`, `ignore_parsing_errors`,
  and `extra_args`: One-to-one mirrors of Faial CLI flags.
- `environment`: Extra environment variables to set when invoking `faial-drf` (string values only).
- Other CLI-mirrored switches: `params`, `block_dim`, `grid_dim`, `only_kernel`, `only_array`, `logic`, `timeout_ms`, `find_true_data_races`, `grid_level`, `all_levels`, `all_dims`, `unreachable`, `ignore_parsing_errors`, and `extra_args`.

Example request payload:

```json
{
  "name": "analyze_kernel",
  "arguments": {
    "source": "__global__ void saxpy(int n, float a, float *x, float *y) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { y[i] = a * x[i] + y[i]; } }",
    "virtual_filename": "saxpy.cu",
    "macros": {
      "RADIUS": "1",
      "LocalBlock": "extern __shared__ unsigned char LocalBlock[]"
    },
    "include_dirs": ["/workspace/includes"],
    "block_dim": "[16,4,1]",
    "grid_dim": "[2,128,1]",
    "ignore_parsing_errors": true
  }
}
```

The response includes Faial's stdout/stderr, parsed JSON output, timing data, per-kernel status summaries, and a concise `human_readable_summary` tailored for LLMs. Each line follows `kernel=<name>;status=<status>;errors=<count>;unknowns=<count>;notes=<tags>`, or `stderr_summary=...` when Faial couldn't parse any kernels.

### Kernel Extraction Helper

Use the built-in helper to extract a kernel snippet and prepare a ready-to-send payload:

```bash
faial-mcp-server extract active_benchmarking/reverseArray_datarace.cu --kernel reverseArray --json
```

This command locates the requested `__global__` function, prints the kernel source (or emits JSON with
`--json`), and replays any launch hints embedded in comments like `//--blockDim=256 --gridDim=1`. Set
`--virtual-filename` if you need a custom logical filename in the output payload.

---

## Docker Usage

### Building the Image

The Dockerfile automatically downloads Faial binaries and sets up the MCP server:

```bash
docker build -t faial-mcp .
```

This creates a self-contained image (~500MB) with:
- Ubuntu 24.04 base
- Faial binaries from the official distribution
- Python 3 and the MCP server package

### Running the Container

#### Network-Based MCP (SSE Transport) - Recommended for Remote Access

```bash
# Run on default port 8000
docker run -p 8000:8000 faial-mcp

# Or customize host/port
docker run -p 9000:9000 faial-mcp faial-mcp-server --transport sse --host 0.0.0.0 --port 9000
```

Connect your MCP client to `http://localhost:8000` (or your custom port).

#### Local Development (stdio Transport)

```bash
docker run -i faial-mcp faial-mcp-server --transport stdio
```

**Note:** stdio transport with Docker requires interactive mode (`-i` flag).

#### Analyzing Files from Host

To analyze files on your host machine, mount a directory into the container:

```bash
# Mount current directory to /workspace in container
docker run -p 8000:8000 -v "$(pwd):/workspace" faial-mcp

# Now the MCP client can reference files like: /workspace/my_kernel.cu
```

On Windows PowerShell:
```powershell
docker run -p 8000:8000 -v "${PWD}:/workspace" faial-mcp
```

### Testing the Server

Once running, test the MCP server with a simple curl request:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "__global__ void add(int *a, int *b) { *a = *b + 1; }",
    "virtual_filename": "test.cu"
  }'
```

### Configuring MCP Clients

#### Claude Desktop

Add to your `claude_desktop_config.json`:

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

#### Cursor / Other Clients

Configure the MCP client to connect to `http://localhost:8000` using SSE transport.

### Environment Variables

Override defaults by passing environment variables to `docker run`:

```bash
docker run -p 8000:8000 \
  -e FAIAL_MCP_TIMEOUT_MS=60000 \
  -e FAIAL_MCP_EXECUTABLE=/opt/faial/faial-drf \
  faial-mcp
```

Available variables:
- `FAIAL_MCP_EXECUTABLE`: Path to `faial-drf` (default: `/opt/faial/faial-drf`)
- `FAIAL_MCP_CU_TO_JSON`: Path to `cu-to-json` helper
- `FAIAL_MCP_TIMEOUT_MS`: Default analysis timeout in milliseconds
- `FAIAL_MCP_HOST`: Default host for network transports (default: `0.0.0.0`)
- `FAIAL_MCP_PORT`: Default port for network transports (default: `8000`)

### Docker Compose (Optional)

For persistent deployment, create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  faial-mcp:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./workspace:/workspace
    environment:
      - FAIAL_MCP_TIMEOUT_MS=60000
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

### Deployment to Cloud

The Docker image can be deployed to any container hosting platform:

#### Docker Hub / Registry

```bash
# Tag and push
docker tag faial-mcp your-registry/faial-mcp:latest
docker push your-registry/faial-mcp:latest
```

#### Cloud Run / ECS / Kubernetes

Deploy the image to your preferred cloud platform, ensuring:
- Port 8000 is exposed
- Container has sufficient memory (recommend 2GB+)
- Set `FAIAL_MCP_TIMEOUT_MS` appropriately for your workload

### Troubleshooting

#### Container exits immediately
Check logs: `docker logs <container-id>`

#### Can't connect to MCP server
- Ensure port is exposed: `docker run -p 8000:8000 ...`
- Check firewall settings
- Verify the server is listening: `docker exec <container-id> netstat -tlnp`

#### Analysis fails with "executable not found"
Verify Faial is in the PATH:
```bash
docker exec <container-id> which faial-drf
docker exec <container-id> faial-drf --version
```

#### Permission errors with mounted volumes
On Linux, you may need to adjust ownership:
```bash
docker run -p 8000:8000 -v "$(pwd):/workspace" --user $(id -u):$(id -g) faial-mcp
```
