# Gemini Workspace Context

This document provides context for the Gemini agent to understand and interact with this project.

## Project Overview

This project is a Model Context Protocol (MCP) server that wraps the Faial CUDA static analysis binaries. It allows Gemini to invoke `faial-drf` through the `analyze_kernel` tool to perform data-race freedom analysis on CUDA and WGSL kernels.

The server is implemented in Python using the `mcp` and `pydantic` libraries. It can be run as a standalone server and communicates over stdio, SSE, or HTTP.

## Building and Running

### Prerequisites

*   Python 3.10+
*   Faial binaries (`faial-drf`, `cu-to-json`) available on the system `PATH`.

### Installation

1.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**

    ```bash
    pip install -e .
    ```

### Running the Server

To run the server, execute the following command:

```bash
faial-mcp-server --transport stdio
```

The server can also be configured to use different transports (`sse`, `streamable-http`) and settings via environment variables.

### Running Linters

This project uses `ruff` for linting. To run the linter, first install the `dev` dependencies:

```bash
pip install -e .[dev]
```

Then, run `ruff`:

```bash
ruff check .
```

## Development Conventions

*   The project follows standard Python packaging conventions.
*   Dependencies are managed in `pyproject.toml`.
*   The main application logic is in `faial_mcp_server/server.py`.
*   The project uses `ruff` for code formatting and linting. Configuration is in the `[tool.ruff]` section of `pyproject.toml`.
