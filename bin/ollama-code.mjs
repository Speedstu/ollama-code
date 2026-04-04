#!/usr/bin/env node

import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import os from "node:os";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { once } from "node:events";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const upstreamCli = path.join(repoRoot, ".upstream", "package", "cli.js");
const shimPath = path.join(repoRoot, "bin", "anthropic-ollama-shim.mjs");
const ollamaHost = (process.env.OLLAMA_HOST || "http://127.0.0.1:11434").replace(/\/+$/, "");
const preferredFastModels = [
  "qwen2.5-coder:1.5b",
  "qwen2.5-coder:3b",
  "qwen2.5-coder:7b",
  "qwen2.5:1.5b",
  "qwen2.5:3b",
  "qwen2.5:7b",
  "deepseek-coder:1.3b",
  "deepseek-coder:6.7b",
  "codellama:7b",
  "llama3.2:3b",
  "phi4-mini",
  "phi3:mini",
];

const ansi = {
  reset: "\x1b[0m",
  blue: "\x1b[38;5;75m",
  boldBlue: "\x1b[1;38;5;75m",
};

function blue(text) {
  return `${ansi.blue}${text}${ansi.reset}`;
}

function boldBlue(text) {
  return `${ansi.boldBlue}${text}${ansi.reset}`;
}

function parseParameterSize(size) {
  if (typeof size !== "string") return Number.POSITIVE_INFINITY;
  const match = size.trim().match(/^(\d+(?:\.\d+)?)([BM])$/i);
  if (!match) return Number.POSITIVE_INFINITY;
  const value = Number(match[1]);
  const unit = match[2].toUpperCase();
  return unit === "B" ? value * 1000 : value;
}

function pickFastLocalModel(models) {
  const installed = models
    .map((model) => ({
      name: typeof model?.name === "string" ? model.name.trim() : "",
      size: parseParameterSize(model?.details?.parameter_size),
      family: typeof model?.details?.family === "string" ? model.details.family : "",
    }))
    .filter((model) => model.name);

  for (const preferred of preferredFastModels) {
    const match = installed.find((model) => model.name === preferred);
    if (match) return match.name;
  }

  const coderCandidates = installed
    .filter((model) => /coder|code|qwen|deepseek|llama|phi/i.test(model.name) || /coder|qwen|deepseek|llama|phi/i.test(model.family))
    .sort((a, b) => a.size - b.size || a.name.localeCompare(b.name));
  if (coderCandidates[0]) return coderCandidates[0].name;

  const anyCandidate = installed.sort((a, b) => a.size - b.size || a.name.localeCompare(b.name));
  return anyCandidate[0]?.name;
}

function formatGiB(bytes) {
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GiB`;
}

function buildModelAdvice(totalMemBytes) {
  const gib = totalMemBytes / 1024 / 1024 / 1024;
  if (gib <= 8) {
    return {
      title: "Low-memory local mode",
      recommendation: "Prefer 0.5b or 1.5b models for responsiveness.",
      installHint: "Try: ollama pull qwen2.5-coder:0.5b",
      recommendedModel: "qwen2.5-coder:0.5b",
    };
  }
  if (gib <= 16) {
    return {
      title: "Balanced local mode",
      recommendation: "Prefer 1.5b or 3b models for a good speed/quality tradeoff.",
      installHint: "Try: ollama pull qwen2.5-coder:1.5b",
      recommendedModel: "qwen2.5-coder:1.5b",
    };
  }
  return {
    title: "Room for larger local models",
    recommendation: "3b or 7b are usually the best balance before latency gets annoying.",
    installHint: "Try: ollama pull qwen2.5-coder:7b",
    recommendedModel: "qwen2.5-coder:7b",
  };
}

function scoreModelForMachine(model, totalMemBytes) {
  const size = parseParameterSize(model?.details?.parameter_size);
  const gib = totalMemBytes / 1024 / 1024 / 1024;
  if (gib <= 8) return size <= 1500 ? 100 : size <= 3000 ? 70 : 10;
  if (gib <= 16) return size <= 3000 ? 100 : size <= 7000 ? 75 : 20;
  return size <= 7000 ? 100 : size <= 15000 ? 80 : 40;
}

function renderModelSelector(models, resolvedModel) {
  const totalMem = os.totalmem();
  const advice = buildModelAdvice(totalMem);
  const lines = [];
  const top = `╭─── ollama-code model selection ───────────────────────────────────────────────╮`;
  const divider = "├───────────────────────────────────────────────────────────────────────────────┤";
  const bottom = "╰───────────────────────────────────────────────────────────────────────────────╯";
  const frameLine = (text) => `${blue("│")} ${text.padEnd(76)}${blue("│")}`;

  lines.push(boldBlue(top));
  lines.push(frameLine(`RAM detected: ${formatGiB(totalMem)}`));
  lines.push(frameLine(advice.title));
  lines.push(frameLine(advice.recommendation));
  lines.push(frameLine(advice.installHint));
  lines.push(blue(divider));

  if (models.length === 0) {
    lines.push(frameLine("No Ollama models installed."));
    lines.push(frameLine(`Recommended first install: ${advice.recommendedModel}`));
    lines.push(frameLine(`Run: ollama pull ${advice.recommendedModel}`));
    lines.push(frameLine("Then relaunch ollama-code."));
    lines.push(boldBlue(bottom));
    return lines.join("\n");
  }

  models
    .slice()
    .sort((a, b) => scoreModelForMachine(b, totalMem) - scoreModelForMachine(a, totalMem) || a.name.localeCompare(b.name))
    .forEach((model, index) => {
      const size = model?.details?.parameter_size || "?";
      const marker = model?.name === resolvedModel ? "●" : " ";
      const recommended = scoreModelForMachine(model, totalMem) >= 100 ? "recommended" : "";
      const label = `${index + 1}. ${model.name} (${size}) ${recommended}`.trim();
      lines.push(`${blue("│")} ${marker === "●" ? boldBlue(marker) : marker} ${label.padEnd(72)}${blue("│")}`);
    });

  lines.push(blue(divider));
  lines.push(frameLine("Enter a number to choose a model, or press Enter to keep the selected one."));
  lines.push(boldBlue(bottom));
  return lines.join("\n");
}

async function maybePromptForModelSelection(argv, models, resolvedModel) {
  const safeModes = new Set(["--help", "-h", "--version", "-v", "-V"]);
  if (argv.length > 0 || argv.some((arg) => safeModes.has(arg))) return resolvedModel;
  if (!process.stdin.isTTY || !process.stdout.isTTY) return resolvedModel;

  process.stdout.write(`${renderModelSelector(models, resolvedModel)}\n`);
  if (models.length === 0) return resolvedModel;

  const ordered = models
    .slice()
    .sort((a, b) => scoreModelForMachine(b, os.totalmem()) - scoreModelForMachine(a, os.totalmem()) || a.name.localeCompare(b.name));

  const rl = createInterface({ input, output });
  try {
    const answer = (await rl.question(`Model [default: ${resolvedModel}] > `)).trim();
    if (!answer) return resolvedModel;
    const index = Number(answer);
    if (Number.isInteger(index) && index >= 1 && index <= ordered.length) {
      return ordered[index - 1].name;
    }
    const matched = ordered.find((model) => model.name === answer);
    return matched?.name || resolvedModel;
  } finally {
    rl.close();
  }
}

async function fetchOllamaCatalog() {
  let response;
  try {
    response = await fetch(`${ollamaHost}/api/tags`);
  } catch (error) {
    throw new Error(
      `Ollama is unreachable at ${ollamaHost}. Start Ollama first, then retry. Original error: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`Ollama preflight failed at ${ollamaHost}/api/tags: ${response.status} ${body}`);
  }

  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(
      `Ollama returned invalid JSON from ${ollamaHost}/api/tags. Original error: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  return Array.isArray(payload?.models) ? payload.models : [];
}

async function resolveOllamaModel(models) {
  if (process.env.OLLAMA_MODEL && process.env.OLLAMA_MODEL.trim()) {
    return process.env.OLLAMA_MODEL.trim();
  }

  const fastModel = pickFastLocalModel(models);
  return fastModel || "qwen2.5-coder:14b";
}

async function ensureBrandPatch(ollamaModel) {
  const markerPath = path.join(repoRoot, ".upstream", ".ollama-brand-patched-v9");
  try {
    const marker = await fs.readFile(markerPath, "utf8");
    if (marker.trim() === ollamaModel.trim()) {
      return;
    }
  } catch {}

  let source = await fs.readFile(upstreamCli, "utf8");
  const replacements = [
    ["(Claude Code)", "(ollama-code)"],
    ["Claude Code", "ollama-code"],
    ["Claude Code - starts an interactive session by default, use -p/--print for", "ollama-code - starts an interactive session by default, use -p/--print for"],
    ["Start the Claude Code MCP server", "Start the ollama-code MCP server"],
    ["Manage Claude Code plugins", "Manage ollama-code plugins"],
    ["Install Claude Code native build. Use [target] to specify version (stable, latest, or specific version)", "Install ollama-code native build. Use [target] to specify version (stable, latest, or specific version)"],
    ["Tip: You can launch Claude Code with just `claude`", "Tip: You can launch ollama-code with just `ollama-code`"],
    ["Usage: claude", "Usage: ollama-code"],
    ["API Usage Billing", "Local Ollama"],
    ["Sonnet 4.6", ollamaModel],
    ["Opus 4.6", ollamaModel],
    ["Welcome back!", "Welcome to ollama-code!"],
    ["return`Welcome back ${q}!`", 'return"Welcome to ollama-code!"'],
    ["Ask Claude to create a new app or clone a repository", "Ask ollama-code to create a new app or clone a repository"],
    ["Run /init to create a CLAUDE.md file with instructions for Claude", "Run /init to create an OLLAMA.md file with instructions for ollama-code"],
    ["Opus now defaults to 1M context · 5x more room, same pricing", "Local Ollama mode · model output speed depends on your machine"],
    ["Use /btw to ask a quick side question without interrupting Claude's current work", "Use /btw to ask a quick side question without interrupting ollama-code's current work"],
    ["Ask Claude to create a todo list when working on complex tasks to track progress and remain on track", "Ask ollama-code to create a todo list when working on complex tasks to track progress and remain on track"],
    ["What should Claude do instead?", "What should ollama-code do instead?"],
    ["Resume with: claude --teleport", "Resume with: ollama-code --teleport"],
    ["claude /logout", "ollama-code /logout"],
    ["claude.ai/code", "ollama-code remote session"],
    ["Connect your local environment for remote-control sessions via claude.ai/code", "Connect your local environment for ollama-code remote sessions"],
    ["Sign in to your Anthropic account", "Manage local/session authentication"],
    ["Use Anthropic Console (API usage billing) instead of Claude subscription", "Use local API-style mode instead of subscription mode"],
    ["Use Claude subscription (default)", "Use local default mode"],
    ["Claude subscription", "local subscription mode"],
    ["Anthropic account", "local account"],
    ["Anthropic Console", "Local Ollama"],
    ["Enable Claude in Chrome integration", "Enable browser integration"],
    ["Disable Claude in Chrome integration", "Disable browser integration"],
    ["CLAUDE.md auto-discovery", "OLLAMA.md auto-discovery"],
    ["when Claude is run with the -p mode", "when ollama-code is run with the -p mode"],
    ["Set up a long-lived authentication token (requires Claude subscription)", "Set up a long-lived authentication token (requires local subscription mode)"],
    ["Model for the current session. Provide an alias for the latest model (e.g. 'sonnet' or 'opus') or a model's full name (e.g. 'claude-sonnet-4-6').", "Model for the current session. Use any Ollama-installed model name, or let ollama-code auto-pick the fastest local model."],
    ["Add an MCP server (stdio or SSE) with a JSON string", "Add an MCP server (stdio or SSE) with a JSON string"],
    ["Import MCP servers from Claude Desktop (Mac and WSL only)", "Import MCP servers from a desktop config (Mac and WSL only)"],
    ["Claude Desktop", "desktop app"],
    ["code.claude.com/docs/en/overview", "github.com/ollama/ollama"],
    ["code.claude.com/docs/en/legal-and-compliance", "github.com/ollama/ollama"],
    ['entrypoint:"claude"', 'entrypoint:"ollama-code"'],
    ['professionalBlue:"rgb(106,155,204)"', 'professionalBlue:"rgb(37,99,235)"'],
    ['chromeYellow:"rgb(251,188,4)"', 'chromeYellow:"rgb(37,99,235)"'],
    ['orange_FOR_SUBAGENTS_ONLY:"rgb(249,115,22)"', 'orange_FOR_SUBAGENTS_ONLY:"rgb(37,99,235)"'],
    ['briefLabelClaude:"rgb(215,119,87)"', 'briefLabelClaude:"rgb(37,99,235)"'],
    ['briefLabelYou:"rgb(37,99,235)"', 'briefLabelYou:"rgb(37,99,235)"'],
    ['selectionBg:"rgb(180, ', 'selectionBg:"rgb(219, '],
    ['process.title="claude"', 'process.title="ollama-code"'],
    ['.name("claude")', '.name("ollama-code")'],
    ['claude:"rgb(215,119,87)"', 'claude:"rgb(37,99,235)"'],
    ['claude:"rgb(255,153,51)"', 'claude:"rgb(37,99,235)"'],
    ['clawd_body:"rgb(215,119,87)"', 'clawd_body:"rgb(37,99,235)"'],
    ['clawd_background:"rgb(0,0,0)"', 'clawd_background:"rgb(255,255,255)"'],
    ['default:{r1L:" ╭",r1E:"─────",r1R:"╮",r2L:" │",r2R:"│ "}', 'default:{r1L:" ▐",r1E:"▛███▜",r1R:"▌",r2L:"▝▜",r2R:"▛▘"}'],
    ['"look-left":{r1L:" ╭",r1E:"─────",r1R:"╮",r2L:" │",r2R:"│ "}', '"look-left":{r1L:" ▐",r1E:"▟███▟",r1R:"▌",r2L:"▝▜",r2R:"▛▘"}'],
    ['"look-right":{r1L:" ╭",r1E:"─────",r1R:"╮",r2L:" │",r2R:"│ "}', '"look-right":{r1L:" ▐",r1E:"▙███▙",r1R:"▌",r2L:"▝▜",r2R:"▛▘"}'],
    ['"arms-up":{r1L:"╭╮",r1E:"─────",r1R:"╭╮",r2L:" │",r2R:"│ "}', '"arms-up":{r1L:"▗▟",r1E:"▛███▜",r1R:"▙▖",r2L:" ▜",r2R:"▛ "}'],
    ['J=Bz.createElement(T,{color:"clawd_body",backgroundColor:"clawd_background"}," OLL ")', 'J=Bz.createElement(T,{color:"clawd_body",backgroundColor:"clawd_background"},"█████")'],
    ['P=Bz.createElement(T,{color:"clawd_body"},"  ","╰────╯","  ")', 'P=Bz.createElement(T,{color:"clawd_body"},"  ","▘▘ ▝▝","  ")'],
    ['z=Bz.createElement(T,{color:"clawd_body"},"╭")', 'z=Bz.createElement(T,{color:"clawd_body"},"▗")'],
    ['$=Bz.createElement(T,{color:"clawd_background",backgroundColor:"clawd_body"}," OLLA ")', '$=Bz.createElement(T,{color:"clawd_background",backgroundColor:"clawd_body"},Y)'],
    ['O=Bz.createElement(T,{color:"clawd_body"},"╮")', 'O=Bz.createElement(T,{color:"clawd_body"},"▖")'],
    ['w=Bz.createElement(T,{backgroundColor:"clawd_body"},"  CODE ")', 'w=Bz.createElement(T,{backgroundColor:"clawd_body"}," ".repeat(7))'],
    ['j=Bz.createElement(T,{color:"clawd_body"},"╰────╯")', 'j=Bz.createElement(T,{color:"clawd_body"},"▘▘ ▝▝")'],
  ];

  for (const [from, to] of replacements) {
    source = source.split(from).join(to);
  }

  source = source
    .replace(/Ask Claude/gi, "Ask ollama-code")
    .replace(/Claude's current work/g, "ollama-code's current work")
    .replace(/Welcome to Claude Code!/g, "Welcome to ollama-code!")
    .replace(/\bCLAUDE\.md\b/g, "OLLAMA.md");

  source = source.replace(/qwen2\.5-coder:\d+(?:\.\d+)?b/gi, ollamaModel);
  source = source.replace(/qwen2\.5:\d+(?:\.\d+)?b/gi, ollamaModel);
  source = source.replace(/deepseek-coder:\d+(?:\.\d+)?b/gi, ollamaModel);
  source = source.replace(/codellama:\d+(?:\.\d+)?b/gi, ollamaModel);
  source = source.replace(/llama3\.2:\d+(?:\.\d+)?b/gi, ollamaModel);

  await fs.writeFile(upstreamCli, source, "utf8");
  await fs.writeFile(markerPath, `${ollamaModel}\n`, "utf8");
}

async function preflightOllama(argv) {
  const safeModes = new Set(["--help", "-h", "--version", "-v", "-V"]);
  if (argv.some((arg) => safeModes.has(arg))) {
    return {
      models: [],
      resolvedModel: (process.env.OLLAMA_MODEL && process.env.OLLAMA_MODEL.trim()) || "local model",
    };
  }

  const models = await fetchOllamaCatalog();
  const resolvedModel = await resolveOllamaModel(models);

  const installedNames = new Set(
    models.flatMap((model) =>
      [model?.name, model?.model]
        .filter((value) => typeof value === "string")
        .map((value) => value.trim()),
    ),
  );

  if (installedNames.size > 0 && !installedNames.has(resolvedModel)) {
    const installed = Array.from(installedNames).sort().join(", ");
    throw new Error(
      `Ollama model \`${resolvedModel}\` is not installed. Installed models: ${installed}`,
    );
  }

  return { models, resolvedModel };
}

function startShim(ollamaModel) {
  const shim = spawn(process.execPath, [shimPath], {
    cwd: repoRoot,
    stdio: ["ignore", "pipe", "inherit"],
    env: {
      ...process.env,
      OLLAMA_MODEL: ollamaModel,
    },
  });

  return new Promise((resolve, reject) => {
    let settled = false;

    shim.once("exit", (code) => {
      if (!settled) {
        reject(new Error(`Shim exited before startup with code ${code ?? "unknown"}`));
      }
    });

    shim.stdout.once("data", (chunk) => {
      settled = true;
      const port = Number(String(chunk).trim());
      if (!Number.isFinite(port) || port <= 0) {
        reject(new Error(`Invalid shim port output: ${String(chunk)}`));
        return;
      }
      resolve({ shim, port });
    });
  });
}

async function main() {
  const { models, resolvedModel } = await preflightOllama(process.argv.slice(2));
  const selectedModel = await maybePromptForModelSelection(process.argv.slice(2), models, resolvedModel);
  if (models.length === 0) {
    const advice = buildModelAdvice(os.totalmem());
    process.stderr.write(`No Ollama models are installed on ${ollamaHost}. Run \`ollama pull ${advice.recommendedModel}\`, then retry.\n`);
    process.exit(1);
  }
  process.env.OLLAMA_MODEL = selectedModel;
  await ensureBrandPatch(selectedModel);
  const { shim, port } = await startShim(selectedModel);

  const child = spawn(process.execPath, [upstreamCli, ...process.argv.slice(2)], {
    cwd: repoRoot,
    stdio: "inherit",
    env: {
      ...process.env,
      ANTHROPIC_BASE_URL: `http://127.0.0.1:${port}`,
      ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY || "ollama-local",
      ANTHROPIC_AUTH_TOKEN: "",
      CLAUDE_CODE_USE_BEDROCK: "",
      CLAUDE_CODE_USE_VERTEX: "",
      CLAUDE_CODE_USE_FOUNDRY: "",
      OLLAMA_MODEL: selectedModel,
      OLLAMA_FAST_LOCAL: process.env.OLLAMA_FAST_LOCAL || "1",
    },
  });

  const shutdown = () => {
    if (!shim.killed) shim.kill("SIGTERM");
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  child.once("exit", (code, signal) => {
    shutdown();
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 0);
  });

  await once(child, "close");
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
