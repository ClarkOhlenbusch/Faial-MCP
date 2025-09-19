from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field, model_validator

FAIAL_EXECUTABLE_ENV = "FAIAL_MCP_EXECUTABLE"
CUTOJSON_ENV = "FAIAL_MCP_CU_TO_JSON"
DEFAULT_TIMEOUT_ENV = "FAIAL_MCP_TIMEOUT_MS"
DEFAULT_TRANSPORT_ENV = "FAIAL_MCP_TRANSPORT"
DEFAULT_HOST_ENV = "FAIAL_MCP_HOST"
DEFAULT_PORT_ENV = "FAIAL_MCP_PORT"

INSTRUCTIONS = (
    "Expose Faial's data-race analysis (`faial-drf`) via the `analyze_kernel` tool. "
    "Provide either `file_path` (an existing CUDA/WGSL file) or inline `source`. "
    "Optional fields mirror CLI flags: `include_dirs`, `macros`, `params`, `block_dim`, "
    "`grid_dim`, `only_kernel`, `only_array`, `timeout_ms`, `ignore_parsing_errors`, "
    "`find_true_data_races`, `grid_level`, `all_levels`, `all_dims`, `unreachable`, and `extra_args`."
)


def _env_path(var: str) -> Optional[Path]:
    value = os.environ.get(var)
    if not value:
        return None
    return Path(value).expanduser()


def _env_int(var: str) -> Optional[int]:
    value = os.environ.get(var)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_str(var: str, default: str) -> str:
    value = os.environ.get(var)
    return value if value else default


def _resolve_with_base(path: Path, base: Optional[Path]) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if base:
        return (base / candidate).resolve()
    return (Path.cwd() / candidate).resolve()


def _resolve_executable(candidate: Path | str) -> str:
    path_candidate = Path(candidate).expanduser()
    if path_candidate.is_file():
        return str(path_candidate)
    found = shutil.which(str(candidate))
    if found:
        return found
    raise FileNotFoundError(f"Could not find executable: {candidate}")


class _ServerDefaults(BaseModel):
    faial_executable: Path = Field(
        default=Path(os.environ.get(FAIAL_EXECUTABLE_ENV, "faial-drf"))
    )
    cu_to_json: Optional[Path] = Field(default=_env_path(CUTOJSON_ENV))
    timeout_ms: Optional[int] = Field(default=_env_int(DEFAULT_TIMEOUT_ENV))

    @classmethod
    def load(cls) -> "_ServerDefaults":
        return cls()


class AnalyzeRequest(BaseModel):
    file_path: Optional[Path] = Field(
        default=None,
        description="Path to a CUDA or WGSL file accessible to the server.",
    )
    source: Optional[str] = Field(
        default=None,
        description="Inline source text to analyze when a file is not available.",
    )
    virtual_filename: Optional[str] = Field(
        default=None,
        description="Filename to associate with inline source (defaults to inline.cu).",
    )
    include_dirs: List[Path] = Field(
        default_factory=list,
        description="Include directories passed via -I/--include-dir.",
    )
    macros: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Macro definitions forwarded via -D.",
    )
    params: Dict[str, int] = Field(
        default_factory=dict,
        description="Kernel parameters forwarded via -p.",
    )
    block_dim: Optional[str] = Field(
        default=None,
        description="Override the block dimension, e.g. [32,1,1].",
    )
    grid_dim: Optional[str] = Field(
        default=None,
        description="Override the grid dimension, e.g. [16,1,1].",
    )
    only_array: Optional[str] = Field(
        default=None,
        description="Restrict analysis to a specific array via --array.",
    )
    only_kernel: Optional[str] = Field(
        default=None,
        description="Restrict analysis to a specific kernel via --kernel.",
    )
    logic: Optional[str] = Field(
        default=None,
        description="Override the SMT logic passed to --logic.",
    )
    timeout_ms: Optional[int] = Field(
        default=None,
        description="Timeout in milliseconds forwarded to --timeout.",
        ge=1,
    )
    faial_executable: Optional[Path] = Field(
        default=None,
        description="Override the faial-drf executable for this request.",
    )
    cu_to_json: Optional[Path] = Field(
        default=None,
        description="Override the cu-to-json helper for this request.",
    )
    ignore_parsing_errors: bool = Field(
        default=False,
        description="Forward --ignore-parsing-errors.",
    )
    find_true_data_races: bool = Field(
        default=False,
        description="Forward --find-true-dr.",
    )
    grid_level: bool = Field(
        default=False,
        description="Forward --grid-level.",
    )
    all_levels: bool = Field(
        default=False,
        description="Forward --all-levels.",
    )
    all_dims: bool = Field(
        default=False,
        description="Forward --all-dims.",
    )
    unreachable: bool = Field(
        default=False,
        description="Forward --unreachable.",
    )
    extra_args: List[str] = Field(
        default_factory=list,
        description="Additional CLI flags passed verbatim after all generated options.",
    )
    working_directory: Optional[Path] = Field(
        default=None,
        description="Working directory to run faial-drf in (defaults to target file's directory).",
    )
    environment: Dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables to use when invoking faial-drf.",
    )

    @model_validator(mode="after")
    def _validate_target(self) -> "AnalyzeRequest":
        if not self.file_path and not self.source:
            raise ValueError("Provide either file_path or source for analysis.")
        return self


class KernelSummary(BaseModel):
    kernel_name: str = Field(description="Kernel identifier reported by Faial.")
    status: str = Field(description="Reported status: drf, racy, or unknown.")
    unknown_count: int = Field(description="Number of proof obligations with unknown outcome.")
    error_count: int = Field(description="Number of concrete counterexamples returned.")


class AnalyzeResponse(BaseModel):
    command: List[str]
    command_string: str
    working_directory: str
    exit_code: int
    timed_out: bool
    duration_seconds: float
    stdout: str
    stderr: str
    stdout_json: Optional[Dict[str, Any]] = None
    stdout_parse_error: Optional[str] = None
    kernel_summaries: List[KernelSummary] = Field(default_factory=list)


async def _run_faial(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    timeout_seconds: Optional[float],
) -> Tuple[int, str, str, bool]:
    process_env = os.environ.copy()
    process_env.update(env)

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=process_env,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Failed to launch faial-drf executable: {command[0]}"
        ) from exc

    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        timed_out = True
        process.kill()
        await process.wait()
        stdout_bytes, stderr_bytes = b"", b""

    exit_code = process.returncode if process.returncode is not None else -1
    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")
    return exit_code, stdout_text, stderr_text, timed_out


def _parse_stdout(stdout_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    stripped = stdout_text.strip()
    if not stripped:
        return None, None

    # First try the whole payload.
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload, None
        return None, "Expected JSON object from faial-drf output."
    except json.JSONDecodeError as exc:
        # Fallback: the output may contain additional lines; try per-line parsing.
        for line in stripped.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload, None
            except json.JSONDecodeError:
                continue
        return None, f"Failed to parse JSON output: {exc}"


def _summaries_from_payload(payload: Dict[str, Any]) -> List[KernelSummary]:
    kernels = payload.get("kernels")
    if not isinstance(kernels, list):
        return []

    summaries: List[KernelSummary] = []
    for entry in kernels:
        if not isinstance(entry, dict):
            continue
        kernel_name = str(entry.get("kernel_name", ""))
        status = str(entry.get("status", "unknown"))
        unknowns = entry.get("unknowns", [])
        errors = entry.get("errors", [])
        unknown_count = len(unknowns) if isinstance(unknowns, list) else 0
        error_count = len(errors) if isinstance(errors, list) else 0
        summaries.append(
            KernelSummary(
                kernel_name=kernel_name,
                status=status,
                unknown_count=unknown_count,
                error_count=error_count,
            )
        )
    return summaries


async def _run_analysis(request: AnalyzeRequest, ctx: Optional[Context] = None) -> AnalyzeResponse:
    defaults = _ServerDefaults.load()
    executable_path = request.faial_executable or defaults.faial_executable
    faial_exec = _resolve_executable(executable_path)

    timeout_ms = request.timeout_ms if request.timeout_ms is not None else defaults.timeout_ms
    timeout_seconds = (timeout_ms / 1000.0) if timeout_ms else None

    cu_to_json = request.cu_to_json if request.cu_to_json is not None else defaults.cu_to_json
    working_directory = request.working_directory.expanduser() if request.working_directory else None

    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        if request.source:
            temp_dir = tempfile.TemporaryDirectory()
            filename = request.virtual_filename or "inline.cu"
            target_path = Path(temp_dir.name) / filename
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(request.source, encoding="utf-8")
            if working_directory is None:
                working_directory = target_path.parent
        else:
            assert request.file_path is not None  # for mypy
            base_dir = working_directory or Path.cwd()
            target_path = _resolve_with_base(request.file_path, base_dir)
            if not target_path.exists():
                raise FileNotFoundError(f"Target file not found: {target_path}")
            if working_directory is None:
                working_directory = target_path.parent

        assert working_directory is not None
        working_directory = working_directory.resolve()

        command: List[str] = [faial_exec, "--json", str(target_path)]
        if timeout_ms:
            command.extend(["--timeout", str(timeout_ms)])
        if cu_to_json:
            cu_path = _resolve_with_base(cu_to_json, working_directory)
            if not cu_path.exists():
                raise FileNotFoundError(f"cu-to-json helper not found: {cu_path}")
            command.extend(["--cu-to-json", str(cu_path)])
        if request.logic:
            command.extend(["--logic", request.logic])
        if request.block_dim:
            command.extend(["--block-dim", request.block_dim])
        if request.grid_dim:
            command.extend(["--grid-dim", request.grid_dim])
        if request.only_array:
            command.extend(["--array", request.only_array])
        if request.only_kernel:
            command.extend(["--kernel", request.only_kernel])
        if request.ignore_parsing_errors:
            command.append("--ignore-parsing-errors")
        if request.find_true_data_races:
            command.append("--find-true-dr")
        if request.grid_level:
            command.append("--grid-level")
        if request.all_levels:
            command.append("--all-levels")
        if request.all_dims:
            command.append("--all-dims")
        if request.unreachable:
            command.append("--unreachable")

        for include_dir in request.include_dirs:
            include_path = _resolve_with_base(include_dir, working_directory)
            command.extend(["-I", str(include_path)])

        for macro, value in request.macros.items():
            if value is None or value == "":
                command.extend(["-D", macro])
            else:
                command.extend(["-D", f"{macro}={value}"])

        for key, value in request.params.items():
            command.extend(["-p", f"{key}={value}"])

        if request.extra_args:
            command.extend(request.extra_args)

        env = {key: str(val) for key, val in request.environment.items()}

        cmd_display = shlex.join(command)
        if ctx is not None:
            await ctx.info(f"Running Faial: {cmd_display}")

        start = time.perf_counter()
        exit_code, stdout_text, stderr_text, timed_out = await _run_faial(
            command,
            working_directory,
            env,
            timeout_seconds,
        )
        duration = time.perf_counter() - start

        payload, parse_error = _parse_stdout(stdout_text)
        summaries = _summaries_from_payload(payload) if payload else []

        if ctx is not None:
            status_fragments = ", ".join(
                f"{summary.kernel_name}:{summary.status}" for summary in summaries
            )
            summary_message = (
                f"Faial completed in {duration:.2f}s (exit={exit_code}, timeout={timed_out})."
            )
            if status_fragments:
                summary_message += f" Kernels: {status_fragments}."
            await ctx.info(summary_message)
            if parse_error:
                await ctx.warning(parse_error)
            if timed_out:
                await ctx.error("Faial command timed out; results may be incomplete.")

        response = AnalyzeResponse(
            command=[str(part) for part in command],
            command_string=cmd_display,
            working_directory=str(working_directory),
            exit_code=exit_code,
            timed_out=timed_out,
            duration_seconds=duration,
            stdout=stdout_text,
            stderr=stderr_text,
            stdout_json=payload,
            stdout_parse_error=parse_error,
            kernel_summaries=summaries,
        )
        return response
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def create_server(*, host: Optional[str] = None, port: Optional[int] = None) -> FastMCP:
    server = FastMCP(
        name="faial-mcp",
        instructions=INSTRUCTIONS,
        host=host or _env_str(DEFAULT_HOST_ENV, "127.0.0.1"),
        port=port or int(_env_str(DEFAULT_PORT_ENV, "8000")),
    )

    @server.tool(name="analyze_kernel", description="Run Faial data-race analysis on CUDA/WGSL code.")
    async def analyze_kernel(request: AnalyzeRequest, ctx: Optional[Context] = None) -> AnalyzeResponse:
        return await _run_analysis(request, ctx)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Faial MCP server entrypoint")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=_env_str(DEFAULT_TRANSPORT_ENV, "stdio"),
        help="Select the MCP transport to use (default: stdio).",
    )
    parser.add_argument(
        "--mount-path",
        default=None,
        help="Mount path for SSE transport (optional).",
    )
    parser.add_argument("--host", default=None, help="Host for SSE/HTTP transports.")
    parser.add_argument("--port", type=int, default=None, help="Port for SSE/HTTP transports.")
    args = parser.parse_args()

    server = create_server(host=args.host, port=args.port)
    server.run(transport=args.transport, mount_path=args.mount_path)


server = create_server()


if __name__ == "__main__":
    main()
