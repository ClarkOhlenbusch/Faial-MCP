# Faial MCP Server

This package wraps the Faial CUDA static analysis binaries in a Model Context Protocol (MCP) server. It allows agents and other MCP clients to invoke `faial-drf` through the `analyze_kernel` tool.

## Prerequisites

- Python 3.10 or newer
- The Faial binaries (`faial-drf`, `cu-to-json`, supporting libraries) available on the host system

## Installation

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

Runs Faial's data-race freedom analysis (`faial-drf`) against a CUDA or WGSL kernel. Supply either `file_path` (server-relative path) or inline `source`. Optional fields map to the CLI flags exposed by Faial (include directories, macros, kernel selection, etc.). The response includes the raw stdout/stderr, parsed JSON payload, timing data, and kernel-level summaries extracted from Faial's JSON UI output.
