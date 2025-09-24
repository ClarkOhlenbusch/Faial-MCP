// @ts-nocheck
import { spawn } from "node:child_process";
import { access } from "node:fs/promises";
import { delimiter } from "node:path";

export type RunResult = {
  stdout: string;
  stderr: string;
  code: number | null;
  timedOut: boolean;
};

export type RunOptions = {
  cwd?: string;
  timeoutMs?: number;
  env?: NodeJS.ProcessEnv;
};

// Reasonable defaults
const DEFAULT_TIMEOUT = 60_000; // 60s

/**
 * Try to resolve the Faial executable path.
 * - Prefer FAIAL_PATH env var
 * - Otherwise rely on the shell PATH to resolve the command name "faial"
 */
export async function resolveFaialExecutable(env: NodeJS.ProcessEnv = process.env): Promise<string> {
  const candidate = env.FAIAL_PATH;
  if (candidate) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // fallthrough
    }
  }
  // On Windows, we'll spawn without explicit path and let the shell PATH find it.
  // We still do a quick PATH check to provide a nicer error if clearly missing.
  const pathVar = env.PATH || env.Path || env.path || "";
  const parts = pathVar.split(delimiter);
  const exeNames = process.platform === "win32" ? ["faial.exe", "faial.cmd", "faial.bat", "faial"] : ["faial"];
  for (const dir of parts) {
    for (const name of exeNames) {
      const candidatePath = `${dir.replace(/\\+$/,'')}\\${name}`;
      try {
        await access(candidatePath);
        return candidatePath;
      } catch {
        // continue
      }
    }
  }
  // Fallback to plain command name; spawn may still find it.
  return process.platform === "win32" ? "faial.exe" : "faial";
}

function getExecutionMode(env?: NodeJS.ProcessEnv): "native" | "wsl" {
  const e = { ...process.env, ...env } as NodeJS.ProcessEnv;
  if (e.FAIAL_MODE === "wsl") return "wsl";
  if (process.platform !== "win32") return "native";
  // On Windows, default to WSL if no obvious native executable is found
  return "wsl";
}

function winPathToWsl(p: string): string {
  // Convert paths like C:\Users\name\file.cu to /mnt/c/Users/name/file.cu
  const m = p.match(/^([A-Za-z]):\\(.*)$/);
  if (!m) {
    // Already POSIX or UNC; do a best-effort backslash replacement
    return p.replace(/\\/g, "/");
  }
  const drive = m[1].toLowerCase();
  const rest = m[2].replace(/\\/g, "/");
  return `/mnt/${drive}/${rest}`;
}

function mapArgsForWsl(args: string[]): string[] {
  if (args.length === 0) return args;
  const mapped = [...args];
  // Heuristics for known path positions
  // case 1: faial analyze <path>
  if (mapped[0] === "analyze" && mapped[1] && !mapped[1].startsWith("-")) {
    mapped[1] = winPathToWsl(mapped[1]);
  }
  // case 2: --dir <dir>
  for (let i = 0; i < mapped.length - 1; i++) {
    if (mapped[i] === "--dir") {
      mapped[i + 1] = winPathToWsl(mapped[i + 1]);
      i++;
    }
  }
  return mapped;
}

export async function runFaial(args: string[], options: RunOptions = {}): Promise<RunResult> {
  const { cwd, timeoutMs = DEFAULT_TIMEOUT, env } = options;
  const mode = getExecutionMode(env);

  if (mode === "wsl") {
    // Build WSL command line
    const wslCmd = "wsl.exe";
    const e = { ...process.env, ...env } as NodeJS.ProcessEnv;
    const distro = e.FAIAL_WSL_DISTRO; // optional
    const faialPath = e.FAIAL_WSL_PATH || "faial"; // path inside WSL
    const wslArgs: string[] = [];
    if (distro) {
      wslArgs.push("-d", distro);
    }
    if (cwd) {
      wslArgs.push("--cd", winPathToWsl(cwd));
    }
    // Separator then the command to run inside WSL
    wslArgs.push("--", faialPath, ...mapArgsForWsl(args));

    return new Promise<RunResult>((resolve) => {
      const child = spawn(wslCmd, wslArgs, {
        env: { ...process.env, ...env },
        stdio: ["ignore", "pipe", "pipe"],
        windowsHide: true,
      });

      let stdout = "";
      let stderr = "";
      let killedByTimeout = false;
      const timer = setTimeout(() => {
        killedByTimeout = true;
        child.kill();
      }, timeoutMs);

      child.stdout?.on("data", (d: Buffer) => (stdout += d.toString()));
      child.stderr?.on("data", (d: Buffer) => (stderr += d.toString()));
      child.on("close", (code) => {
        clearTimeout(timer);
        resolve({ stdout: stdout.trim(), stderr: stderr.trim(), code, timedOut: killedByTimeout });
      });
      child.on("error", (err) => {
        clearTimeout(timer);
        resolve({ stdout: "", stderr: String(err), code: -1, timedOut: false });
      });
    });
  }

  // Native (Linux/macOS or Windows if native executable available)
  const command = await resolveFaialExecutable(env);
  return new Promise<RunResult>((resolve) => {
    const child = spawn(command, args, {
      cwd,
      env: { ...process.env, ...env },
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });

    let stdout = "";
    let stderr = "";
    let killedByTimeout = false;
    const timer = setTimeout(() => {
      killedByTimeout = true;
      child.kill();
    }, timeoutMs);

    child.stdout?.on("data", (d: Buffer) => (stdout += d.toString()));
    child.stderr?.on("data", (d: Buffer) => (stderr += d.toString()));
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ stdout: stdout.trim(), stderr: stderr.trim(), code, timedOut: killedByTimeout });
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      resolve({ stdout: "", stderr: String(err), code: -1, timedOut: false });
    });
  });
}

export async function getFaialVersion(): Promise<RunResult> {
  return runFaial(["--version"], { timeoutMs: 10_000 });
}
