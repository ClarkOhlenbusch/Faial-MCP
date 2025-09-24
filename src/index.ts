// @ts-nocheck
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { getFaialVersion, runFaial } from "./faialRunner.js";
import express from "express";
import cors from "cors";
import multer from "multer";
import crypto from "crypto";

const server = new McpServer({
  name: "faial-mcp",
  version: "0.1.0",
});

// Shared schemas
const AnalyzeFileSchema = {
  path: z.string().describe("Path to a single .cu/.cuh/.cpp/.h file to analyze"),
  cwd: z.string().optional().describe("Working directory for analysis (optional)"),
  extraArgs: z
    .array(z.string())
    .optional()
    .describe("Additional CLI flags to pass through to Faial"),
  timeoutMs: z.number().int().positive().optional().describe("Timeout in milliseconds"),
};

const AnalyzeDirectorySchema = {
  dir: z.string().describe("Directory containing CUDA sources to analyze"),
  pattern: z.string().optional().describe("Glob or pattern for files (tool-side filtering)"),
  recursive: z.boolean().optional().describe("Recurse into subdirectories"),
  extraArgs: z.array(z.string()).optional().describe("Additional CLI flags for Faial"),
  timeoutMs: z.number().int().positive().optional().describe("Timeout in milliseconds"),
};

const AnalyzeTextSchema = {
  code: z.string().describe("Raw CUDA/C++ source code to analyze (written to a temp file)"),
  filename: z.string().optional().describe("Virtual filename to help Faial (e.g., main.cu)"),
  extraArgs: z.array(z.string()).optional().describe("Additional CLI flags for Faial"),
  timeoutMs: z.number().int().positive().optional().describe("Timeout in milliseconds"),
};

server.registerTool(
  "faial.get_version",
  {
    title: "Faial: Get Version",
    description: "Return Faial CLI version to verify installation",
    inputSchema: {},
  },
  async () => {
    const res = await getFaialVersion();
    const text = res.timedOut
      ? "Timed out calling Faial --version"
      : res.code === 0
      ? res.stdout || res.stderr || "Faial version returned no output"
      : `Failed to get version (code ${res.code}).\n${res.stderr || res.stdout}`;
    return {
      content: [{ type: "text", text }],
      isError: res.code !== 0 || res.timedOut,
    };
  }
);

server.registerTool(
  "faial.analyze_file",
  {
    title: "Faial: Analyze File",
    description: "Analyze a single CUDA/C++ source file with Faial",
    inputSchema: AnalyzeFileSchema,
  },
  async ({ path, cwd, extraArgs, timeoutMs }) => {
    // Assumption: Faial supports calling like: faial analyze <path> --format json
    const args = ["analyze", path, "--format", "json", ...(extraArgs || [])];
    const res = await runFaial(args, { cwd, timeoutMs });
    return normalizeFaialResult(res, `Analysis for ${path}`);
  }
);

server.registerTool(
  "faial.analyze_directory",
  {
    title: "Faial: Analyze Directory",
    description: "Analyze all matching files in a directory with Faial",
    inputSchema: AnalyzeDirectorySchema,
  },
  async ({ dir, pattern, recursive, extraArgs, timeoutMs }) => {
    // Assumption: Faial supports directory analysis: faial analyze --dir <dir> --recursive --format json
    const args = ["analyze", "--dir", dir];
    if (recursive) args.push("--recursive");
    if (pattern) args.push("--pattern", pattern);
    args.push("--format", "json");
    if (extraArgs) args.push(...extraArgs);
    const res = await runFaial(args, { timeoutMs });
    return normalizeFaialResult(res, `Directory analysis for ${dir}`);
  }
);

server.registerTool(
  "faial.analyze_text",
  {
    title: "Faial: Analyze Text",
    description: "Analyze raw source code by writing to a temp file and invoking Faial",
    inputSchema: AnalyzeTextSchema,
  },
  async ({ code, filename = "snippet.cu", extraArgs, timeoutMs }) => {
    // Minimal temp-file flow to avoid adding runtime deps; OS temp dir
    const os = await import("node:os");
    const fs = await import("node:fs/promises");
    const pathMod = await import("node:path");
    const tmpDir = await fs.mkdtemp(pathMod.join(os.tmpdir(), "faial-"));
    const filePath = pathMod.join(tmpDir, filename);
    await fs.writeFile(filePath, code, "utf8");
    try {
      const args = ["analyze", filePath, "--format", "json", ...(extraArgs || [])];
      const res = await runFaial(args, { timeoutMs });
      return normalizeFaialResult(res, `Analysis for in-memory ${filename}`);
    } finally {
      // Best effort cleanup
      try { await fs.unlink(filePath); } catch {}
      try { await fs.rmdir(tmpDir); } catch {}
    }
  }
);

function normalizeFaialResult(res: any, label: string) {
  // Prefer JSON output when available; fall back to text
  const output = res.stdout || res.stderr;
  let text = output || `${label}: no output`;
  let content: any[] = [];
  try {
    const json = JSON.parse(output);
    content = [{ type: "text", text: JSON.stringify(json, null, 2) }];
  } catch {
    content = [{ type: "text", text }];
  }
  const isError = res.timedOut || (res.code ?? 1) !== 0;
  return { content, isError };
}

// HTTP Server implementation
const app = express();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit for uploaded files
    files: 1 // Only one file at a time
  },
  fileFilter: (req, file, cb) => {
    // Only allow CUDA/C++ files
    const allowedTypes = [
      'text/x-cuda',
      'text/x-c++src',
      'text/x-csrc',
      'text/x-chdr',
      'text/x-c++hdr',
      'application/octet-stream'
    ];

    // Check file extension as fallback
    const allowedExts = ['.cu', '.cuh', '.cpp', '.c', '.h', '.hpp', '.cxx', '.cc'];
    const ext = file.originalname.toLowerCase().substring(file.originalname.lastIndexOf('.'));

    if (allowedTypes.includes(file.mimetype) || allowedExts.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error('Only CUDA and C++ files are allowed'));
    }
  }
});

// Simple rate limiting (in production, consider using express-rate-limit)
const rateLimitMap = new Map();
const RATE_LIMIT_WINDOW = 60000; // 1 minute
const RATE_LIMIT_MAX_REQUESTS = 10; // Max 10 requests per minute per IP

// Rate limiting function
function checkRateLimit(ip: string): boolean {
  const now = Date.now();
  const windowStart = now - RATE_LIMIT_WINDOW;

  if (!rateLimitMap.has(ip)) {
    rateLimitMap.set(ip, []);
  }

  const requests = rateLimitMap.get(ip) || [];
  const recentRequests = requests.filter((timestamp: number) => timestamp > windowStart);

  if (recentRequests.length >= RATE_LIMIT_MAX_REQUESTS) {
    return false; // Rate limit exceeded
  }

  recentRequests.push(now);
  rateLimitMap.set(ip, recentRequests);
  return true;
}

// Cleanup old rate limit entries periodically
setInterval(() => {
  const now = Date.now();
  const windowStart = now - RATE_LIMIT_WINDOW;
  for (const [ip, requests] of rateLimitMap.entries()) {
    const validRequests = (requests as number[]).filter((timestamp: number) => timestamp > windowStart);
    if (validRequests.length === 0) {
      rateLimitMap.delete(ip);
    } else {
      rateLimitMap.set(ip, validRequests);
    }
  }
}, RATE_LIMIT_WINDOW);

// Security middleware
app.use((req, res, next) => {
  // Security headers
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  next();
});

// Middleware
app.use(cors({
  origin: process.env.NODE_ENV === 'production'
    ? ['https://cursor.sh', 'https://claude.ai', 'https://modelcontextprotocol.io']
    : true, // Allow all origins in development
  credentials: true
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'faial-mcp', version: '0.1.0' });
});

// MCP SSE endpoint
app.get('/sse', async (req, res) => {
  try {
    // Set up SSE headers
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Headers', 'Cache-Control');

    const transport = new SSEServerTransport("/messages", res);
    await server.connect(transport);

    // Handle client disconnections
    req.on('close', () => {
      transport.close();
    });

    req.on('error', (err) => {
      console.error('SSE request error:', err);
      transport.close();
    });
  } catch (error) {
    console.error('SSE endpoint error:', error);
    res.status(500).json({ error: 'SSE connection failed' });
  }
});

// MCP messages endpoint
app.post('/messages', async (req, res) => {
  await server.processMessage(req.body);
  res.json({});
});

// File upload endpoints for easier file analysis
app.post('/analyze/upload', upload.single('file'), async (req, res) => {
  // Rate limiting
  const clientIP = req.ip || req.connection.remoteAddress || 'unknown';
  if (!checkRateLimit(clientIP)) {
    return res.status(429).json({ error: 'Rate limit exceeded. Please try again later.' });
  }

  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // Write uploaded file to temp location
    const os = await import("node:os");
    const fs = await import("node:fs/promises");
    const pathMod = await import("node:path");
    const tmpDir = await fs.mkdtemp(pathMod.join(os.tmpdir(), "faial-upload-"));
    const filePath = pathMod.join(tmpDir, req.file.originalname);
    await fs.writeFile(filePath, req.file.buffer);

    const extraArgs = req.body.extraArgs ? JSON.parse(req.body.extraArgs) : [];
    const timeoutMs = req.body.timeoutMs ? parseInt(req.body.timeoutMs) : undefined;

    try {
      const args = ["analyze", filePath, "--format", "json", ...extraArgs];
      const result = await runFaial(args, { timeoutMs });

      const output = result.stdout || result.stderr || 'No output';
      let analysisResult;
      try {
        analysisResult = JSON.parse(output);
      } catch {
        analysisResult = { rawOutput: output };
      }

      res.json({
        success: !result.timedOut && result.code === 0,
        result: analysisResult,
        metadata: {
          filename: req.file.originalname,
          size: req.file.size,
          timedOut: result.timedOut,
          exitCode: result.code
        }
      });
    } finally {
      // Cleanup
      try { await fs.unlink(filePath); } catch {}
      try { await fs.rmdir(tmpDir); } catch {}
    }
  } catch (error) {
    res.status(500).json({ error: 'Analysis failed', details: String(error) });
  }
});

// Text analysis endpoint
app.post('/analyze/text', async (req, res) => {
  // Rate limiting
  const clientIP = req.ip || req.connection.remoteAddress || 'unknown';
  if (!checkRateLimit(clientIP)) {
    return res.status(429).json({ error: 'Rate limit exceeded. Please try again later.' });
  }

  try {
    const { code, filename = 'snippet.cu', extraArgs = [], timeoutMs } = req.body;

    if (!code) {
      return res.status(400).json({ error: 'No code provided' });
    }

    // Write code to temp file
    const os = await import("node:os");
    const fs = await import("node:fs/promises");
    const pathMod = await import("node:path");
    const tmpDir = await fs.mkdtemp(pathMod.join(os.tmpdir(), "faial-text-"));
    const filePath = pathMod.join(tmpDir, filename);
    await fs.writeFile(filePath, code, "utf8");

    try {
      const args = ["analyze", filePath, "--format", "json", ...extraArgs];
      const result = await runFaial(args, { timeoutMs: timeoutMs || 60000 });

      const output = result.stdout || result.stderr || 'No output';
      let analysisResult;
      try {
        analysisResult = JSON.parse(output);
      } catch {
        analysisResult = { rawOutput: output };
      }

      res.json({
        success: !result.timedOut && result.code === 0,
        result: analysisResult,
        metadata: {
          filename,
          timedOut: result.timedOut,
          exitCode: result.code
        }
      });
    } finally {
      // Cleanup
      try { await fs.unlink(filePath); } catch {}
      try { await fs.rmdir(tmpDir); } catch {}
    }
  } catch (error) {
    res.status(500).json({ error: 'Analysis failed', details: String(error) });
  }
});

// Main function to determine transport mode
async function main() {
  const mode = process.env.TRANSPORT_MODE || 'http';
  const port = parseInt(process.env.PORT || '3000');

  if (mode === 'stdio') {
    // Original stdio mode for local development
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.log('Faial MCP server running in stdio mode');
  } else {
    // HTTP mode for hosting
    app.listen(port, () => {
      console.log(`Faial MCP server running in HTTP mode on port ${port}`);
      console.log(`SSE endpoint: http://localhost:${port}/sse`);
      console.log(`Health check: http://localhost:${port}/health`);
    });
  }
}

main().catch((err) => {
  console.error("Faial MCP server failed:", err);
  process.exit(1);
});
