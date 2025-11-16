# TODO: This file defines the MCP server for Faial. The following changes are proposed to improve the usability of the Faial tool for the agent:
# 1.  **Improve the `INSTRUCTIONS`:** The `INSTRUCTIONS` should be updated to provide more context on Faial's compositional analysis and the importance of providing self-contained, well-formed kernel snippets.
# 2.  **Enhance `AnalyzeRequest` descriptions:** The descriptions for the fields in the `AnalyzeRequest` model should be expanded to provide more detailed guidance on how to use each parameter effectively.
# 3.  **Add examples:** Consider adding examples of "good" and "bad" kernel snippets to the documentation or as part of the tool definition to help the agent learn how to prepare code for Faial analysis.

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field, ValidationError, model_validator

# TODO: The following functions and constants are all related to handling environment variables and should be moved to a new file called `env.py`.
# {{{- env
FAIAL_EXECUTABLE_ENV = "FAIAL_MCP_EXECUTABLE"
CUTOJSON_ENV = "FAIAL_MCP_CU_TO_JSON"
DEFAULT_TIMEOUT_ENV = "FAIAL_MCP_TIMEOUT_MS"
DEFAULT_TRANSPORT_ENV = "FAIAL_MCP_TRANSPORT"
DEFAULT_HOST_ENV = "FAIAL_MCP_HOST"
DEFAULT_PORT_ENV = "FAIAL_MCP_PORT"
DEBUG_REQUESTS_ENV = "FAIAL_MCP_DEBUG_REQUESTS"
EXTRACT_COMMAND = "extract"
# }}}


INSTRUCTIONS = (
    "Expose Faial's data-race analysis (`faial-drf`) via the `analyze_kernel` tool. "
    "\n\n"
    "## Input Method\n"
    "**MANDATORY:** Provide kernel code via the `source` parameter only. File paths are not supported to ensure reliable, "
    "container-compatible analysis and proper input preparation.\n"
    "\n"
    "- **`source` (required)**: Pass self-contained kernel code directly as a string. Include all necessary type definitions, "
    "macros, and dependencies. Do not reference host-only globals or headers that are not provided via the request." 
    " Example: `{\"source\": \"__global__ void kernel() { ... }\", \"virtual_filename\": \"test.cu\"}`\n"
    "- **`macros`**: Provide a dictionary mapping macro names to optional values (e.g., `{\"DEBUG\": \"1\", \"FLAG\": null}`). "
    "CLI-style strings such as `\"DEBUG 1\"` or `\"FLAG=value\"` are also accepted and automatically normalized.\n"
    "\n"
    "## Critical Requirements\n"
    "- **MANDATORY:** Provide **self-contained, well-formed kernel snippets** - each kernel must be isolated with all dependencies\n"
    "- **MANDATORY:** Include all necessary type definitions, structs, constants, and macros within the source\n"
    "- **MANDATORY:** Avoid full-file dumps - analyze individual kernels separately to prevent false positives\n"
    "- Use `include_dirs` and `macros` to resolve external dependencies when needed."
    " `include_dirs` are resolved relative to `working_directory` (defaults to the server's current"
    " directory), so provide absolute paths or set `working_directory` when referencing mounted"
    " headers.\n"
    "- Use `working_directory` to point Faial at the root containing headers and helper binaries"
    " (default: server launch directory).\n"
    "- Use `ignore_parsing_errors` flag sparingly for partial code analysis\n"
    "\n"
    "## Example Snippets\n"
    "- **Good:** Self-contained code with helper types/macros.\n"
    "```cuda\n"
    "#define TILE 128\n"
    "struct Pair { float a; float b; };\n"
    "__device__ inline float add_pair(Pair p) { return p.a + p.b; }\n"
    "__global__ void sum_pairs(Pair *data, float *out, int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n) { out[i] = add_pair(data[i]); }\n"
    "}\n"
    "```\n"
    "- **Bad:** References undefined helpers or host-only globals.\n"
    "```cuda\n"
    "__global__ void sum_pairs(Pair *data) {\n"
    "    out[idx] = add_pair(data[idx]);  // Pair, add_pair, out undefined\n"
    "}\n"
    "```\n"
    "\n"
    "## Configuration Parameters\n"
    "Optional fields mirror CLI flags: `include_dirs`, `macros`, `params`, `block_dim`, "
    "`grid_dim`, `only_kernel`, `only_array`, `timeout_ms`, `ignore_parsing_errors`, "
    "`find_true_data_races`, `grid_level`, `all_levels`, `all_dims`, `unreachable`, and `extra_args`.\n"
    "- **Output control:** Responses default to concise summaries plus short excerpts. Set"
    " `include_raw_output=true` to receive the full stdout/stderr/JSON payload when needed.\n"
    "\n"
    "## Agent Preparation Guidelines\n"
    "- **Isolate each kernel:** Extract individual kernels from source files\n"
    "- **Resolve dependencies:** Include all required types, constants, and function declarations\n"
    "- **Avoid cross-contamination:** Never analyze multiple unrelated kernels together\n"
    "- **Test incrementally:** Start with minimal, well-formed kernels before complex analysis\n"
    "- **Use helper tooling:** Run `faial-mcp-server extract <file> --kernel <name> --json` to generate ready-to-send payloads."
)

# TODO: The following functions are all related to handling environment variables and should be moved to a new file called `env.py`.
# {{{- env
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


def _env_bool(var: str, default: bool = False) -> bool:
    value = os.environ.get(var)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default
# }}}



# TODO: The following functions are general-purpose utility functions and should be moved to a new file called `util.py`.
# {{{- util
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


def _truncate_text(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
# }}}



# Extraction helpers
# {{{- extract
def _normalize_dim_hint(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return cleaned
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return cleaned

    tokens = re.split(r"[xX,\s]+", cleaned)
    tokens = [token for token in tokens if token]
    if not tokens:
        return cleaned
    if all(token.isdigit() for token in tokens):
        while len(tokens) < 3:
            tokens.append("1")
        tokens = tokens[:3]
        return f"[{tokens[0]},{tokens[1]},{tokens[2]}]"
    return cleaned


def _parse_launch_hints(text: str) -> Dict[str, str]:
    hints: Dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("//--"):
            continue
        payload = stripped[2:].strip()
        for chunk in payload.split():
            entry = chunk.strip()
            if not entry:
                continue
            if entry.startswith("--"):
                entry = entry[2:]
            if "=" not in entry:
                continue
            key, value = entry.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                continue
            normalized_key = key.lower()
            if normalized_key == "blockdim" and "block_dim" not in hints:
                hints["block_dim"] = _normalize_dim_hint(value)
            elif normalized_key == "griddim" and "grid_dim" not in hints:
                hints["grid_dim"] = _normalize_dim_hint(value)
    return hints


def _line_start_index(text: str, index: int) -> int:
    newline = text.rfind("\n", 0, index)
    return newline + 1 if newline != -1 else 0


def _skip_string(text: str, start_index: int, quote: str) -> int:
    i = start_index + 1
    escaped = False
    while i < len(text):
        char = text[i]
        if char == "\\" and not escaped:
            escaped = True
            i += 1
            continue
        if char == quote and not escaped:
            return i + 1
        escaped = False
        i += 1
    return len(text)


def _skip_line_comment(text: str, start_index: int) -> int:
    newline = text.find("\n", start_index)
    return len(text) if newline == -1 else newline + 1


def _skip_block_comment(text: str, start_index: int) -> int:
    end = text.find("*/", start_index + 2)
    return len(text) if end == -1 else end + 2


def _find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    i = open_index
    while i < len(text):
        char = text[i]
        if char == '{':
            depth += 1
            i += 1
            continue
        if char == '}':
            depth -= 1
            i += 1
            if depth == 0:
                return i
            continue
        if char == '"':
            i = _skip_string(text, i, '"')
            continue
        if char == "'":
            i = _skip_string(text, i, "'")
            continue
        if char == '/' and i + 1 < len(text):
            next_char = text[i + 1]
            if next_char == '/':
                i = _skip_line_comment(text, i)
                continue
            if next_char == '*':
                i = _skip_block_comment(text, i)
                continue
        i += 1
    raise ValueError("Unmatched braces in kernel definition")


def _collect_kernel_snippets(text: str) -> List[Dict[str, Any]]:
    kernels: List[Dict[str, Any]] = []
    search_pos = 0
    while True:
        idx = text.find("__global__", search_pos)
        if idx == -1:
            break
        search_pos = idx + len("__global__")
        brace_idx = text.find("{", idx)
        if brace_idx == -1:
            break
        header = text[idx:brace_idx]
        param_start = header.rfind("(")
        if param_start == -1:
            continue
        before_param = header[:param_start]
        name_match = re.search(r"([A-Za-z_][\w]*)\s*$", before_param)
        if not name_match:
            continue
        kernel_name = name_match.group(1)
        start = _line_start_index(text, idx)
        try:
            end = _find_matching_brace(text, brace_idx)
        except ValueError:
            continue
        snippet = text[start:end].rstrip() + "\n"
        line_number = text.count("\n", 0, start) + 1
        kernels.append({
            "name": kernel_name,
            "source": snippet,
            "line": line_number,
        })
    return kernels


def _run_extract_command(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(
        prog=f"faial-mcp-server {EXTRACT_COMMAND}",
        description="Extract a __global__ kernel from a CUDA file and emit a ready-to-use payload.",
    )
    parser.add_argument("file", type=Path, help="Path to the CUDA source file")
    parser.add_argument(
        "-k",
        "--kernel",
        help="Kernel name to extract (required when multiple kernels exist).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only the analyze_kernel arguments as JSON.",
    )
    parser.add_argument(
        "--virtual-filename",
        help="Override the virtual_filename used in the output payload (defaults to the source filename).",
    )
    args = parser.parse_args(argv)

    file_path = args.file.expanduser()
    if not file_path.is_file():
        parser.error(f"File not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    kernels = _collect_kernel_snippets(text)
    if not kernels:
        parser.error("No __global__ kernels found in the supplied file.")

    selected: Optional[Dict[str, Any]] = None
    if args.kernel:
        for entry in kernels:
            if entry["name"] == args.kernel:
                selected = entry
                break
        if selected is None:
            available = ", ".join(kernel["name"] for kernel in kernels)
            parser.error(f"Kernel '{args.kernel}' not found. Available kernels: {available}")
    else:
        if len(kernels) > 1:
            available = ", ".join(kernel["name"] for kernel in kernels)
            parser.error(
                "Multiple kernels detected. Specify --kernel to choose one. Available kernels: "
                f"{available}"
            )
        selected = kernels[0]

    assert selected is not None
    hints = _parse_launch_hints(text)
    payload: Dict[str, Any] = {
        "source": selected["source"],
        "virtual_filename": args.virtual_filename or file_path.name,
    }
    if "block_dim" in hints:
        payload["block_dim"] = hints["block_dim"]
    if "grid_dim" in hints:
        payload["grid_dim"] = hints["grid_dim"]

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(
        f"Kernel '{selected['name']}' (line {selected['line']}) extracted from {file_path}.",
        file=sys.stdout,
    )
    print(selected["source"], end="" if selected["source"].endswith("\n") else "\n", file=sys.stdout)
    print(file=sys.stdout)
    if hints:
        print("Inferred launch hints:", file=sys.stdout)
        for key, value in hints.items():
            print(f"  {key}={value}", file=sys.stdout)
        print(file=sys.stdout)
    sample_request = {
        "name": "analyze_kernel",
        "arguments": payload,
    }
    print("Sample analyze_kernel request payload:", file=sys.stdout)
    print(json.dumps(sample_request, indent=2), file=sys.stdout)


def _should_run_extract() -> bool:
    return len(sys.argv) > 1 and sys.argv[1] == EXTRACT_COMMAND
# }}}



# TODO: The following Pydantic models are all related to the data structures used in the server. They should be moved to a new file called `models.py`.
# {{{- models
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
    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        if "file_path" in values:
            raise ValueError(
                "`file_path` is not supported by the Faial MCP server. "
                "Provide inline kernel code via the `source` field instead."
            )

        macros = values.get("macros")
        if isinstance(macros, list):
            normalized: Dict[str, Optional[str]] = {}
            for entry in macros:
                if isinstance(entry, str):
                    candidate = entry.strip()
                    if not candidate:
                        continue
                    if "=" in candidate:
                        name, remainder = candidate.split("=", 1)
                        normalized[name.strip()] = remainder.strip() or None
                    else:
                        parts = candidate.split(None, 1)
                        name = parts[0]
                        value = parts[1] if len(parts) > 1 else None
                        normalized[name.strip()] = value.strip() if value else None
                elif isinstance(entry, dict):
                    for key, value in entry.items():
                        normalized[str(key)] = None if value is None else str(value)
            values["macros"] = normalized

        return values

    source: str = Field(
        description=(
            "MANDATORY: Self-contained kernel source code as a string. Include all necessary type definitions, "
            "macros, structs, constants, and dependencies. Each kernel must be isolated and complete."
        ),
    )
    virtual_filename: Optional[str] = Field(
        default=None,
        description=(
            "Filename to associate with inline source (defaults to inline.cu). "
            "Use this to specify the file extension (.cu for CUDA, .wgsl for WGSL) when using `source`."
        ),
    )
    include_dirs: List[Path] = Field(
        default_factory=list,
        description=(
            "Include directories for resolving #include directives, passed via -I/--include-dir. "
            "Use this to provide paths to necessary header files required for analysis."
        ),
    )
    macros: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description=(
            "Macro definitions forwarded via -D (e.g., {'DEBUG': '1', 'FEATURE': None}). "
            "Use this to define preprocessor macros required for successful kernel parsing."
        ),
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
    include_raw_output: bool = Field(
        default=False,
        description=(
            "Set to true to include raw stdout/stderr/stdout_json in the response."
            " By default the server returns only concise summaries plus short excerpts"
            " to keep payloads small."
        ),
    )

    @model_validator(mode="after")
    def _validate_source(self) -> "AnalyzeRequest":
        if not self.source or not self.source.strip():
            raise ValueError("source parameter is mandatory and cannot be empty. Provide self-contained kernel code as a string.")
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
    human_readable_summary: Optional[str] = Field(
        default=None,
        description=(
            "Concise, agent-friendly status summary. Each line follows "
            "`kernel=<name>;status=<status>;errors=<count>;unknowns=<count>;notes=<tags>` "
            "or a single `stderr_summary=...` entry when no kernels parsed."
        ),
    )
    stdout_excerpt: Optional[str] = Field(
        default=None,
        description="Truncated stdout preview when summary-only mode suppresses the full text.",
    )
    stderr_excerpt: Optional[str] = Field(
        default=None,
        description="Truncated stderr preview when summary-only mode suppresses the full text.",
    )
# }}}



# TODO: The following functions are all part of the core analysis logic and should be moved to a new file called `faial.py`.
# {{{- faial
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


def _format_human_summary(
    summaries: List[KernelSummary], stderr_text: str, max_lines: int = 3
) -> Optional[str]:
    if summaries:
        lines: List[str] = []
        for summary in summaries:
            notes = []
            if summary.error_count:
                notes.append("counterexamples")
            if summary.unknown_count:
                notes.append("unknown_obligations")
            if not notes:
                notes.append("no_issues")
            lines.append(
                "kernel={name};status={status};errors={errors};unknowns={unknowns};notes={notes}".format(
                    name=summary.kernel_name or "<unnamed>",
                    status=summary.status.lower(),
                    errors=summary.error_count,
                    unknowns=summary.unknown_count,
                    notes=",".join(notes),
                )
            )
        return "\n".join(lines)

    stderr_lines = [line.strip() for line in stderr_text.strip().splitlines() if line.strip()]
    if not stderr_lines:
        return None

    truncated = False
    if len(stderr_lines) > max_lines:
        stderr_lines = stderr_lines[:max_lines]
        truncated = True
    summary = " | ".join(stderr_lines)
    if truncated:
        summary += " | ..."
    return f"stderr_summary={summary}"


async def _run_analysis(request: AnalyzeRequest, ctx: Optional[Context] = None) -> AnalyzeResponse:
    defaults = _ServerDefaults.load()
    executable_path = request.faial_executable or defaults.faial_executable
    faial_exec = _resolve_executable(executable_path)

    timeout_ms = request.timeout_ms if request.timeout_ms is not None else defaults.timeout_ms
    timeout_seconds = (timeout_ms / 1000.0) if timeout_ms else None

    cu_to_json = request.cu_to_json if request.cu_to_json is not None else defaults.cu_to_json
    working_directory = request.working_directory.expanduser() if request.working_directory else None
    include_base = (
        request.working_directory.expanduser().resolve()
        if request.working_directory
        else Path.cwd()
    )

    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        # Only source-based analysis is supported
        temp_dir = tempfile.TemporaryDirectory()
        filename = request.virtual_filename or "inline.cu"
        target_path = Path(temp_dir.name) / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(request.source, encoding="utf-8")
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
            include_path = _resolve_with_base(include_dir, include_base)
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
        human_summary = _format_human_summary(summaries, stderr_text)

        stdout_excerpt = _truncate_text(stdout_text)
        stderr_excerpt = _truncate_text(stderr_text)
        response_stdout_json = None
        response_stdout = ""
        response_stderr = ""

        if request.include_raw_output:
            response_stdout = stdout_text
            response_stderr = stderr_text
            response_stdout_json = payload
            stdout_excerpt = None
            stderr_excerpt = None

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
            if human_summary:
                await ctx.info(f"Summary:\n{human_summary}")
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
            stdout=response_stdout,
            stderr=response_stderr,
            stdout_json=response_stdout_json,
            stdout_parse_error=parse_error,
            kernel_summaries=summaries,
            human_readable_summary=human_summary,
            stdout_excerpt=stdout_excerpt,
            stderr_excerpt=stderr_excerpt,
        )
        return response
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
# }}}



def create_server(*, host: Optional[str] = None, port: Optional[int] = None) -> FastMCP:
    server = FastMCP(
        name="faial-mcp",
        instructions=INSTRUCTIONS,
        host=host or _env_str(DEFAULT_HOST_ENV, "127.0.0.1"),
        port=port or int(_env_str(DEFAULT_PORT_ENV, "8000")),
    )

    @server.tool(
        name="analyze_kernel",
        description=(
            "Run Faial data-race analysis on isolated CUDA/WGSL kernels to detect potential data races. "
            "**MANDATORY:** Use ONLY the `source` parameter - file paths are not supported. "
            "Faial uses formal verification to prove kernels are data-race free or provide concrete counterexamples. "
            "**CRITICAL:** Provide self-contained, well-formed kernel snippets with ALL necessary dependencies included. "
            "Each kernel must be analyzed separately to avoid false positives from cross-kernel interference. "
            "Include all type definitions, structs, constants, and macros within the source code. "
            "Use `include_dirs` and `macros` to resolve external dependencies when needed. "
            "Set `ignore_parsing_errors=True` only for partial code analysis."
        )
    )
    async def analyze_kernel(request: dict, ctx: Optional[Context] = None) -> AnalyzeResponse:
        # Handle both wrapped {"request": {...}} and flattened {...} parameter formats for compatibility
        try:
            if "request" in request:
                # Standard MCP format
                req_obj = AnalyzeRequest(**request["request"])
            else:
                # Flattened format (compatibility)
                req_obj = AnalyzeRequest(**request)
        except ValidationError as exc:
            pieces = []
            for error in exc.errors():
                location = ".".join(str(part) for part in error.get("loc", ())) or "input"
                pieces.append(f"{location}: {error.get('msg')}")
            message = "; ".join(pieces) or "Invalid analyze_kernel request."
            raise ValueError(message) from exc

        if _env_bool(DEBUG_REQUESTS_ENV):
            print(
                f"DEBUG: Received request parameters: {json.dumps(req_obj.model_dump(), indent=2)}",
                file=sys.stderr,
            )
        return await _run_analysis(req_obj, ctx)

    return server


def main() -> None:
    if _should_run_extract():
        _run_extract_command(sys.argv[2:])
        return
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
