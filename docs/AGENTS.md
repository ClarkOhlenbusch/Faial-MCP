# Repository Guidelines

## Project Structure & Modules
- `faial_mcp_server/`: Python package. Main entrypoint in `server.py` exposes the MCP tools and CLI.
- `pyproject.toml`: Package metadata, entry points, and tooling config (Ruff, line-length 100).
- `README.md`: Setup, run instructions, and environment variables.
- Optional local env: `.venv/` for development. No committed tests directory yet.

## Build, Test, and Development
- Install (dev): `pip install -e .[dev]`
  - Installs the package and dev tools (Ruff).
- Run server: `faial-mcp-server --transport stdio`
  - Use `--transport sse` or `--transport streamable-http` for networked transports.
- Lint: `ruff check .`
  - Add `--fix` to auto-apply safe fixes.
- Format: `ruff format .`

## Coding Style & Naming
- Python ≥ 3.10 with type hints; prefer `pydantic` models for request/response schemas.
- Line length 100 (see `pyproject.toml`). Indentation: 4 spaces.
- Names: modules and functions `snake_case`, classes `CapWords`.
- Keep the public API in `faial_mcp_server.server`; add tools via `@server.tool(...)` and typed `BaseModel` payloads.

## Testing Guidelines
- No test suite is included yet. If adding tests:
  - Location: `tests/` mirrored to package structure (e.g., `tests/test_server.py`).
  - Framework: `pytest` recommended; add to `project.optional-dependencies.dev`.
  - Run: `pytest -q` and aim for coverage on tool wiring and CLI argument parsing.

## Commit & Pull Requests
- Commits: Use imperative, concise subjects (≤ 72 chars). Example: `feat: add grid-level flag handling`.
- Include context in the body when changing tool semantics or CLI.
- PRs: Provide description, linked issues, repro/usage snippet, and screenshots/logs for failures. Note any env vars required.

## Security & Configuration
- Binaries: Ensure `faial-drf` is on `PATH` or set `FAIAL_MCP_EXECUTABLE`.
- Helpers: Optionally set `FAIAL_MCP_CU_TO_JSON`.
- Timeouts/transport: `FAIAL_MCP_TIMEOUT_MS`, `FAIAL_MCP_TRANSPORT`, `FAIAL_MCP_HOST`, `FAIAL_MCP_PORT`.
- Example: `FAIAL_MCP_EXECUTABLE=/opt/faial/bin/faial-drf faial-mcp-server --transport stdio`.

