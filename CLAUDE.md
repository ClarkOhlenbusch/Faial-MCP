# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Faial MCP Server wraps Faial (a CUDA static analysis tool) via the Model Context Protocol (MCP), enabling AI agents to detect data races in GPU kernels. The evaluation goal is to assess LLMs' abilities to fix data-race bugs detected by Faial in CUDA kernels from `/saved_example_kernels`. Each fixable issue requires one strategically placed `__syncthreads()` call.

### Key Problem Domain
- **Data-race detection**: Faial identifies potential concurrent memory access violations in CUDA kernels
- **Race-freedom proof**: Faial either proves DRF (data-race free) or provides counterexamples showing racy behavior
- **Synchronization fixes**: Solutions involve adding `__syncthreads()` barriers at critical points to enforce thread synchronization

## Development Commands

### Local Setup
```bash
# Create and activate virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS/WSL
# .venv\Scripts\activate   # Windows

# Install the package
pip install -e .

# Install development dependencies (linter/formatter)
pip install -e .[dev]
```

### Running the Server
```bash
# Default: stdio transport (for local agent integration)
faial-mcp-server --transport stdio

# SSE transport (for remote/HTTP clients)
faial-mcp-server --transport sse --host 0.0.0.0 --port 8000

# With custom Faial executable or cu-to-json helper
FAIAL_MCP_EXECUTABLE=/path/to/faial-drf faial-mcp-server --transport stdio
```

### Code Quality
```bash
# Lint with ruff
ruff check faial_mcp_server/

# Format code
ruff format faial_mcp_server/
```

### Docker Operations
```bash
# Build Docker image (recommended for deployment)
docker build -t faial-mcp .

# Run with SSE transport (default port 8000)
docker run -p 8000:8000 faial-mcp

# Run with stdio transport (for local testing)
docker run -i faial-mcp faial-mcp-server --transport stdio

# Mount workspace for kernel file access
docker run -p 8000:8000 -v "$(pwd):/workspace" faial-mcp

# Test the server with curl
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"source": "__global__ void add(int *a, int *b) { *a = *b + 1; }", "virtual_filename": "test.cu"}'
```

## Architecture & Code Structure

### Package Layout
- `faial_mcp_server/server.py`: Main MCP server implementation (~540 lines)
  - `INSTRUCTIONS`: Prompt guidance for agents on correct kernel submission
  - `AnalyzeRequest`: Pydantic model defining the `analyze_kernel` tool parameters
  - `AnalyzeResponse`: Response model with stdout, parsed JSON, kernel summaries, exit codes
  - `_run_faial()`: Async subprocess executor for `faial-drf` binary
  - `_parse_stdout()`: JSON parser for Faial's JSON output format
  - `_summaries_from_payload()`: Extracts kernel DRF status and error counts from JSON
  - `create_server()`: FastMCP factory; defines the `analyze_kernel` tool
  - `main()`: CLI entrypoint; handles transport selection (stdio/SSE/HTTP)

### Design Patterns
- **Compositional input**: Faial accepts only self-contained, well-formed kernel snippets via `source` (not file paths)
- **Async execution**: Uses `asyncio` for subprocess management with timeout support
- **Environment variables** for configuration:
  - `FAIAL_MCP_EXECUTABLE`: Path to `faial-drf` binary (auto-detected from PATH if unset)
  - `FAIAL_MCP_CU_TO_JSON`: Path to helper utility (optional)
  - `FAIAL_MCP_TIMEOUT_MS`: Default analysis timeout in milliseconds
  - `FAIAL_MCP_TRANSPORT`, `FAIAL_MCP_HOST`, `FAIAL_MCP_PORT`: Server defaults
- **Temp file handling**: Source code is written to a temporary directory, analyzed, then cleaned up

### The `analyze_kernel` Tool
**Purpose**: Run Faial's data-race freedom analysis on isolated CUDA/WGSL kernels.

**Input (`AnalyzeRequest` fields)**:
- `source` (required): Self-contained kernel code as a string with all dependencies included
- `virtual_filename`: For inline source, specifies file extension (.cu/.wgsl); defaults to "inline.cu"
- `include_dirs`: Directories for resolving #include directives (-I flags)
- `macros`: Preprocessor macro definitions (-D flags)
- `block_dim`, `grid_dim`: Override execution configuration (e.g., "[32,1,1]")
- `only_kernel`, `only_array`: Restrict analysis to specific kernel or array
- `ignore_parsing_errors`: Allow partial code analysis (avoid unless necessary)
- `find_true_data_races`: Flag for stricter data-race detection
- `grid_level`, `all_levels`, `all_dims`, `unreachable`: Additional analysis modes
- `timeout_ms`: Override default timeout per request
- `faial_executable`, `cu_to_json`: Override defaults for this request
- `working_directory`, `environment`: Custom execution context

**Output (`AnalyzeResponse` fields)**:
- `command`, `command_string`: Invoked Faial command and its shell representation
- `exit_code`, `timed_out`, `duration_seconds`: Execution details
- `stdout`, `stderr`: Raw Faial output
- `stdout_json`: Parsed JSON payload (if present)
- `kernel_summaries`: List of `KernelSummary` objects with kernel name, status (drf/racy/unknown), error/unknown counts

## Critical Kernel Submission Guidelines

From `Faial-Optimal-Usage-Findings.md` and embedded `INSTRUCTIONS`:

1. **Provide self-contained code only**: Include all type definitions, structs, constants, macros, and helper functions within the `source` string
2. **Avoid file paths**: The `file_path` parameter is not supported; always use the `source` parameter
3. **Isolate kernels**: Analyze individual kernels separately to prevent false positives from cross-kernel interference
4. **Ensure syntactic correctness**: CUDA code must be well-formed for Faial to parse it
5. **Use include_dirs and macros** for external dependencies rather than full-file dumps
6. **Invoke `analyze_kernel` only once per kernel** before fixing and re-testing

## Example Kernel Locations

Test kernels with known data races are in `saved_example_kernels/`:
- `example.cu`: Basic example
- `bitonicSort_datarace.cu`: Bitonic sort with race condition
- `convolution2D_datarace.cu`: 2D convolution with synchronization bug
- `matrixMul_datarace.cu`, `reduction_datarace.cu`: Other multi-threaded patterns

Each typically has one or more data races fixable by adding `__syncthreads()` at the correct location(s).

## Environment Configuration

### For Local Faial
If Faial binaries are not on PATH, set:
```bash
export FAIAL_MCP_EXECUTABLE=/path/to/faial-drf
export FAIAL_MCP_CU_TO_JSON=/path/to/cu-to-json  # if needed
```

### For Docker Deployment
- Default `FAIAL_MCP_EXECUTABLE` in Docker image: `/opt/faial/faial-drf`
- Override via `docker run -e FAIAL_MCP_EXECUTABLE=...`

### Timeout Tuning
- Default: None (inherits from Faial's internal default or system limits)
- For longer analyses: `FAIAL_MCP_TIMEOUT_MS=120000` (2 minutes)
- Per-request override via `analyze_kernel` parameter

## Key Dependencies

- **mcp >= 1.14.0**: Model Context Protocol library (FastMCP framework)
- **pydantic >= 2.5**: Data validation and schema definition
- **Python >= 3.10**: Minimum version required
- **faial-drf binary**: Faial analysis engine (must be available on PATH or via env var)

## Integration Points

### MCP Client Setup
Clients (Claude Desktop, Cursor, etc.) connect to the server and invoke `analyze_kernel`:

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

### Subprocess Isolation
Faial runs as a separate process with its own stdin/stdout/stderr. Temporary files are cleaned up automatically.

### JSON Output Format
Faial's `--json` flag produces a structured output parsed by `_parse_stdout()`. The JSON object contains a `kernels` array with entries like:
```json
{
  "kernels": [
    {
      "kernel_name": "mykernel",
      "status": "racy",
      "errors": [
        { /* counterexample trace */ }
      ],
      "unknowns": []
    }
  ]
}
```

## Testing Strategy

1. **Unit-level**: Manually invoke `analyze_kernel` with example kernels from `saved_example_kernels/`
2. **Docker**: Use test scripts (`test-docker.sh`, `test-docker.ps1`) to verify the container builds and runs
3. **Integration**: Wire up an MCP client (Claude Desktop, Cursor) and test end-to-end kernel analysis

## Performance Notes

- **Analysis time**: Depends on kernel complexity and Faial solver; typically seconds to minutes
- **Memory**: Faial may require 1-2GB for complex analyses; Docker image defaults to container limits
- **Timeout**: Set appropriately for your workload (60-120 seconds for typical kernels)

## Common Troubleshooting

- **"Could not find executable"**: Ensure `faial-drf` is on PATH or set `FAIAL_MCP_EXECUTABLE`
- **"source parameter is mandatory"**: Always provide `source` as a string, not `file_path`
- **Parsing errors**: Verify kernel code is syntactically correct and self-contained
- **Timeout**: Increase `FAIAL_MCP_TIMEOUT_MS` for complex kernels
- **Docker connection refused**: Check port mapping (`docker run -p 8000:8000`) and firewall

## Refactoring Opportunities

The `server.py` file contains TODOs suggesting future modularization:
- Move env var handling to `env.py`
- Extract utility functions to `util.py`
- Move Pydantic models to `models.py`
- Move analysis logic to `faial.py`

These changes are not urgent but would improve maintainability for larger projects.
