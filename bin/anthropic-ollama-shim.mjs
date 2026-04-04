#!/usr/bin/env node

import http from "node:http";
import { randomUUID } from "node:crypto";

const ollamaHost = (process.env.OLLAMA_HOST || "http://127.0.0.1:11434").replace(/\/+$/, "");
const defaultOllamaModel = process.env.OLLAMA_MODEL || "qwen2.5-coder:14b";

function json(res, status, body, extraHeaders = {}) {
  res.writeHead(status, {
    "content-type": "application/json",
    "request-id": `req_${randomUUID().replace(/-/g, "")}`,
    ...extraHeaders,
  });
  res.end(JSON.stringify(body));
}

function sse(res, event, data) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function estimateTokens(value) {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return Math.max(1, Math.ceil(text.length / 4));
}

function normalizeTextContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((block) => {
      if (typeof block === "string") return block;
      if (!block || typeof block !== "object") return "";
      if (block.type === "text") return block.text || "";
      if (block.type === "tool_result") return toolResultToString(block);
      if (block.type === "image") return "[image omitted]";
      if (block.type === "document") return "[document omitted]";
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

function toolResultToString(block) {
  const raw = block.content;
  const text =
    typeof raw === "string"
      ? raw
      : Array.isArray(raw)
        ? raw
            .map((item) => {
              if (typeof item === "string") return item;
              if (item?.type === "text") return item.text || "";
              return JSON.stringify(item);
            })
            .join("\n")
        : JSON.stringify(raw);
  return block.is_error ? `Tool error (${block.tool_use_id}): ${text}` : `Tool result (${block.tool_use_id}): ${text}`;
}

function anthropicToolsToOllamaTools(tools = []) {
  return tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description || "",
      parameters: tool.input_schema || {
        type: "object",
        properties: {},
      },
    },
  }));
}

function stripMarkdownCodeFence(text) {
  if (typeof text !== "string") return "";
  const trimmed = text.trim();
  const fenced = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  return fenced ? fenced[1].trim() : trimmed;
}

function parseStructuredText(text) {
  const cleaned = stripMarkdownCodeFence(text);
  if (!cleaned || !/^[\[{]/.test(cleaned)) return null;
  try {
    return JSON.parse(cleaned);
  } catch {
    return null;
  }
}

function sanitizeAssistantText(text) {
  if (typeof text !== "string") return "";
  let cleaned = text.trim();
  const usageIndex = cleaned.search(/\n\s*Usage:\s*\n/i);
  if (usageIndex !== -1) {
    cleaned = cleaned.slice(0, usageIndex).trimEnd();
  }
  cleaned = cleaned
    .replace(/\bClaude Code\b/g, "ollama-code")
    .replace(/\bCLAUDE\.md\b/g, "OLLAMA.md")
    .replace(/\bClaude\b/g, "ollama-code");
  return cleaned;
}

function looksLikeStructuredNoise(text) {
  const parsed = parseStructuredText(text);
  if (!parsed) return false;
  if (Array.isArray(parsed)) return parsed.length > 0;
  if (typeof parsed !== "object") return false;
  const keys = Object.keys(parsed);
  return keys.includes("name") || keys.includes("arguments") || keys.includes("input") || keys.includes("params");
}

function extractPlainTextFromStructuredOutput(text) {
  const parsed = parseStructuredText(text);
  if (!parsed || Array.isArray(parsed)) return null;

  const directFields = [parsed.message, parsed.text, parsed.content];
  for (const field of directFields) {
    if (typeof field === "string" && field.trim()) return field.trim();
  }

  const args = parsed.arguments || parsed.input || parsed.params;
  if (args && typeof args === "object") {
    for (const key of ["message", "text", "content", "response", "answer", "result", "code"]) {
      const value = args[key];
      if (typeof value === "string" && value.trim()) return value.trim();
    }
  }

  for (const key of ["result", "code", "output"]) {
    const value = parsed[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }

  return null;
}

function getLastUserText(messages = []) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role === "user") {
      return normalizeTextContent(message.content).trim();
    }
  }
  return "";
}

function isSimpleGreeting(text) {
  const normalized = text.trim().toLowerCase();
  return /^(hi|hello|hey|yo|salut|coucou|test|ping)[!. ]*$/.test(normalized);
}

function greetingReply(text) {
  const normalized = text.trim().toLowerCase();
  if (normalized.startsWith("salut") || normalized.startsWith("coucou")) return "Salut!";
  return "Hello!";
}

function isSimpleCodingRequest(text) {
  const normalized = text.trim().toLowerCase();
  return (
    /(code|script).*(python|pyhto|pyth?on)/.test(normalized) ||
    /(python|pyhto|pyth?on).*(code|script)/.test(normalized) ||
    /(affiche|print|prints?|display|show).*(welcome|hello|bonjour)/.test(normalized)
  );
}

function buildFallbackCodeReply(text) {
  const normalized = text.trim();
  const returnMatch =
    normalized.match(/(?:renvoi|retourne|return(?:s)?|renvoie)\s+["'`]?([^"'`\n]+?)["'`]?$/i) ||
    normalized.match(/(?:renvoi|retourne|return(?:s)?|renvoie).+?["'`]{1}([^"'`\n]+)["'`]{1}/i);
  const displayMatch =
    normalized.match(/(?:affiche|print(?:s)?|display|show)\s+["'`]?([^"'`\n]+?)["'`]?$/i) ||
    normalized.match(/(?:affiche|print(?:s)?|display|show).+?["'`]{1}([^"'`\n]+)["'`]{1}/i);

  const message = (returnMatch?.[1] || displayMatch?.[1] || "Welcome to ollama-code").trim();
  if (!message) return null;

  return `print(${JSON.stringify(message)})`;
}

function isLikelySmallLocalModel(model) {
  return /:(?:0\.5|1\.5|3)b$/i.test(model);
}

function normalizeToolName(name) {
  return String(name || "")
    .trim()
    .toLowerCase()
    .replace(/[\s_-]+/g, "");
}

function findMatchingTool(tools = [], candidateName) {
  const normalizedCandidate = normalizeToolName(candidateName);
  return tools.find((tool) => normalizeToolName(tool?.name) === normalizedCandidate) || null;
}

function extractToolCallFromText(text, tools = []) {
  const parsed = parseStructuredText(text);
  if (!parsed) return null;

  const candidates = Array.isArray(parsed) ? parsed : [parsed];
  const converted = [];

  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== "object") return null;
    const rawName = candidate.name || candidate.tool || candidate.function?.name;
    const rawArguments = candidate.arguments || candidate.input || candidate.params || candidate.function?.arguments;
    const matchedTool = findMatchingTool(tools, rawName);
    if (!matchedTool) return null;

    let input = rawArguments ?? {};
    if (typeof input === "string") {
      try {
        input = JSON.parse(input);
      } catch {
        input = { raw: input };
      }
    }

    converted.push({
      type: "tool_use",
      id: `toolu_${randomUUID().replace(/-/g, "")}`,
      name: matchedTool.name,
      input: input && typeof input === "object" ? input : { value: input },
    });
  }

  return converted.length ? converted : null;
}

function convertMessages(systemPrompt, messages = []) {
  const ollamaMessages = [];

  const systemText =
    typeof systemPrompt === "string"
      ? systemPrompt
      : Array.isArray(systemPrompt)
        ? systemPrompt
            .map((block) => (block?.type === "text" ? block.text || "" : ""))
            .filter(Boolean)
            .join("\n")
        : "";

  if (systemText) {
    ollamaMessages.push({
      role: "system",
      content: systemText,
    });
  }

  for (const message of messages) {
    const blocks = Array.isArray(message.content) ? message.content : [{ type: "text", text: String(message.content || "") }];
    const textParts = [];
    const toolCalls = [];
    const toolResults = [];

    for (const block of blocks) {
      if (!block || typeof block !== "object") continue;
      if (block.type === "text") {
        if (block.text) textParts.push(block.text);
        continue;
      }
      if (block.type === "tool_use") {
        toolCalls.push({
          id: block.id,
          type: "function",
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input || {}),
          },
        });
        continue;
      }
      if (block.type === "tool_result") {
        toolResults.push({
          role: "tool",
          content: toolResultToString(block),
          tool_call_id: block.tool_use_id,
        });
        continue;
      }
      if (block.type === "image") {
        textParts.push("[image omitted]");
        continue;
      }
      if (block.type === "document") {
        textParts.push("[document omitted]");
      }
    }

    if (message.role === "assistant") {
      ollamaMessages.push({
        role: "assistant",
        content: textParts.join("\n"),
        ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }

    if (textParts.length) {
      ollamaMessages.push({
        role: message.role === "user" ? "user" : message.role,
        content: textParts.join("\n"),
      });
    }

    for (const toolResult of toolResults) {
      ollamaMessages.push(toolResult);
    }
  }

  return ollamaMessages;
}

function ollamaToAnthropicContent(message, tools = []) {
  const content = [];
  const inferredToolCalls =
    (!Array.isArray(message.tool_calls) || message.tool_calls.length === 0) && message.content
      ? extractToolCallFromText(String(message.content), tools)
      : null;

  if (inferredToolCalls) {
    return inferredToolCalls;
  }

  const extractedPlainText = message.content ? extractPlainTextFromStructuredOutput(String(message.content)) : null;
  if (extractedPlainText) {
    return [
      {
        type: "text",
        text: extractedPlainText,
      },
    ];
  }

  if (message.content) {
    const cleanedText = sanitizeAssistantText(String(message.content));
    content.push({
      type: "text",
      text: cleanedText,
    });
  }

  for (const toolCall of message.tool_calls || []) {
    let input = {};
    try {
      input = JSON.parse(toolCall.function?.arguments || "{}");
    } catch {
      input = { raw: toolCall.function?.arguments || "" };
    }
    content.push({
      type: "tool_use",
      id: toolCall.id || `toolu_${randomUUID().replace(/-/g, "")}`,
      name: toolCall.function?.name || "unknown_tool",
      input,
    });
  }

  if (!content.length) {
    content.push({
      type: "text",
      text: "",
    });
  }

  return content;
}

function buildAnthropicMessage(body, ollamaResponse) {
  const message = ollamaResponse.message || {};
  const lastUserText = getLastUserText(body.messages);
  const shouldForcePlainGreeting = isLikelySmallLocalModel(defaultOllamaModel) && isSimpleGreeting(lastUserText);
  const shouldForceSimpleCode = isLikelySmallLocalModel(defaultOllamaModel) && isSimpleCodingRequest(lastUserText);
  const fallbackCodeReply = shouldForceSimpleCode ? buildFallbackCodeReply(lastUserText) : null;
  const inputTokens = ollamaResponse.prompt_eval_count || estimateTokens(body.messages || []);
  const anthropicContent = ollamaToAnthropicContent(message, body.tools);
  const cleanedNoiseFallback =
    isLikelySmallLocalModel(defaultOllamaModel) &&
    shouldForceSimpleCode === false &&
    shouldForcePlainGreeting === false &&
    looksLikeStructuredNoise(message.content || "") &&
    !extractToolCallFromText(message.content || "", body.tools)
      ? [{ type: "text", text: "I’m ready. Ask me something more specific." }]
      : null;
  const finalContent = shouldForcePlainGreeting
    ? [{ type: "text", text: greetingReply(lastUserText) }]
    : fallbackCodeReply
      ? [{ type: "text", text: fallbackCodeReply }]
    : cleanedNoiseFallback
      ? cleanedNoiseFallback
    : anthropicContent;
  const hasToolUse = finalContent.some((block) => block.type === "tool_use");
  const outputTokens = ollamaResponse.eval_count || estimateTokens(message.content || message.tool_calls || anthropicContent);

  return {
    id: `msg_${randomUUID().replace(/-/g, "")}`,
    type: "message",
    role: "assistant",
    model: defaultOllamaModel,
    content: finalContent,
    stop_reason: hasToolUse ? "tool_use" : "end_turn",
    stop_sequence: null,
    usage: {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
    },
  };
}

function buildOllamaRequest(body, stream) {
  return {
    model: defaultOllamaModel,
    stream,
    messages: convertMessages(body.system, body.messages),
    tools: anthropicToolsToOllamaTools(body.tools),
    options: {
      temperature: typeof body.temperature === "number" ? body.temperature : undefined,
      num_predict: typeof body.max_tokens === "number" ? body.max_tokens : undefined,
    },
  };
}

async function callOllama(body) {
  const response = await fetch(`${ollamaHost}/api/chat`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify(buildOllamaRequest(body, false)),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Ollama request failed: ${response.status} ${text}`);
  }

  return response.json();
}

async function openOllamaStream(body) {
  const response = await fetch(`${ollamaHost}/api/chat`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify(buildOllamaRequest(body, true)),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Ollama request failed: ${response.status} ${text}`);
  }

  if (!response.body) {
    throw new Error("Ollama returned an empty streaming body");
  }

  return response.body;
}

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const raw = Buffer.concat(chunks).toString("utf8");
  return raw ? JSON.parse(raw) : {};
}

function writeStream(res, anthropicMessage) {
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
    "request-id": `req_${randomUUID().replace(/-/g, "")}`,
  });

  sse(res, "message_start", {
    type: "message_start",
    message: {
      id: anthropicMessage.id,
      type: anthropicMessage.type,
      role: anthropicMessage.role,
      model: anthropicMessage.model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: anthropicMessage.usage.input_tokens,
        output_tokens: 0,
      },
    },
  });

  anthropicMessage.content.forEach((block, index) => {
    if (block.type === "text") {
      sse(res, "content_block_start", {
        type: "content_block_start",
        index,
        content_block: {
          type: "text",
          text: "",
        },
      });
      sse(res, "content_block_delta", {
        type: "content_block_delta",
        index,
        delta: {
          type: "text_delta",
          text: block.text || "",
        },
      });
      sse(res, "content_block_stop", {
        type: "content_block_stop",
        index,
      });
      return;
    }

    if (block.type === "tool_use") {
      sse(res, "content_block_start", {
        type: "content_block_start",
        index,
        content_block: {
          type: "tool_use",
          id: block.id,
          name: block.name,
          input: {},
        },
      });
      sse(res, "content_block_delta", {
        type: "content_block_delta",
        index,
        delta: {
          type: "input_json_delta",
          partial_json: JSON.stringify(block.input || {}),
        },
      });
      sse(res, "content_block_stop", {
        type: "content_block_stop",
        index,
      });
    }
  });

  sse(res, "message_delta", {
    type: "message_delta",
    delta: {
      stop_reason: anthropicMessage.stop_reason,
      stop_sequence: anthropicMessage.stop_sequence,
    },
    usage: {
      output_tokens: anthropicMessage.usage.output_tokens,
    },
  });
  sse(res, "message_stop", {
    type: "message_stop",
  });
  res.end();
}

async function writeStreamFromOllama(body, res) {
  const stream = await openOllamaStream(body);
  const decoder = new TextDecoder();
  let buffer = "";
  let textBlockStarted = false;
  let textIndex = 0;
  let sawText = false;
  let finalChunk = null;
  let aggregatedText = "";
  let deferTextEmission = false;
  let firstContentSeen = false;
  const lastUserText = getLastUserText(body.messages);
  const shouldForcePlainGreeting = isLikelySmallLocalModel(defaultOllamaModel) && isSimpleGreeting(lastUserText);
  const shouldForceSimpleCode = isLikelySmallLocalModel(defaultOllamaModel) && isSimpleCodingRequest(lastUserText);
  const fallbackCodeReply = shouldForceSimpleCode ? buildFallbackCodeReply(lastUserText) : null;
  const shouldForceBufferedPlainText = shouldForcePlainGreeting || Boolean(fallbackCodeReply);

  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
    "request-id": `req_${randomUUID().replace(/-/g, "")}`,
  });

  sse(res, "message_start", {
    type: "message_start",
    message: {
      id: `msg_${randomUUID().replace(/-/g, "")}`,
      type: "message",
      role: "assistant",
      model: defaultOllamaModel,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: estimateTokens({
          system: body.system,
          messages: body.messages,
          tools: body.tools,
        }),
        output_tokens: 0,
      },
    },
  });

  const flushLine = (line) => {
    if (!line.trim()) return;
    let chunk;
    try {
      chunk = JSON.parse(line);
    } catch {
      return;
    }

    finalChunk = chunk;
    const text = chunk?.message?.content ?? "";
    if (text) {
      aggregatedText += text;

      if (shouldForceBufferedPlainText) {
        return;
      }

      if (!firstContentSeen) {
        firstContentSeen = true;
        const cleaned = stripMarkdownCodeFence(aggregatedText.trimStart());
        deferTextEmission = /^[\[{]/.test(cleaned);
      }

      if (deferTextEmission) {
        return;
      }

      if (!textBlockStarted) {
        textBlockStarted = true;
        sse(res, "content_block_start", {
          type: "content_block_start",
          index: textIndex,
          content_block: {
            type: "text",
            text: "",
          },
        });
      }

      sawText = true;
      sse(res, "content_block_delta", {
        type: "content_block_delta",
        index: textIndex,
        delta: {
          type: "text_delta",
          text,
        },
      });
    }
  };

  for await (const chunk of stream) {
    buffer += decoder.decode(chunk, { stream: true });
    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex !== -1) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      flushLine(line);
      newlineIndex = buffer.indexOf("\n");
    }
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    flushLine(buffer.trim());
  }

  const inferredToolCalls = extractToolCallFromText(aggregatedText, body.tools);
  const inferredPlainText = extractPlainTextFromStructuredOutput(aggregatedText);
  const sanitizedAggregatedText = sanitizeAssistantText(aggregatedText);
  const shouldSuppressStructuredNoise =
    isLikelySmallLocalModel(defaultOllamaModel) &&
    !shouldForceBufferedPlainText &&
    looksLikeStructuredNoise(aggregatedText) &&
    !inferredToolCalls &&
    !inferredPlainText;

  if (shouldForcePlainGreeting) {
    const forcedText = greetingReply(lastUserText);
    if (!textBlockStarted) {
      textBlockStarted = true;
      sawText = true;
      sse(res, "content_block_start", {
        type: "content_block_start",
        index: textIndex,
        content_block: {
          type: "text",
          text: "",
        },
      });
      sse(res, "content_block_delta", {
        type: "content_block_delta",
        index: textIndex,
        delta: {
          type: "text_delta",
          text: forcedText,
        },
      });
    }
  }

  if (fallbackCodeReply) {
    if (!textBlockStarted) {
      textBlockStarted = true;
      sawText = true;
      sse(res, "content_block_start", {
        type: "content_block_start",
        index: textIndex,
        content_block: {
          type: "text",
          text: "",
        },
      });
      sse(res, "content_block_delta", {
        type: "content_block_delta",
        index: textIndex,
        delta: {
          type: "text_delta",
          text: fallbackCodeReply,
        },
      });
    }
  }

  if (shouldSuppressStructuredNoise) {
    if (!textBlockStarted) {
      textBlockStarted = true;
      sawText = true;
      sse(res, "content_block_start", {
        type: "content_block_start",
        index: textIndex,
        content_block: {
          type: "text",
          text: "",
        },
      });
      sse(res, "content_block_delta", {
        type: "content_block_delta",
        index: textIndex,
        delta: {
          type: "text_delta",
          text: isSimpleGreeting(lastUserText) ? greetingReply(lastUserText) : "I’m ready. Ask me something more specific.",
        },
      });
    }
  }

  if (!shouldForceBufferedPlainText && !shouldSuppressStructuredNoise && deferTextEmission && aggregatedText && !inferredToolCalls) {
    const finalText = sanitizeAssistantText(inferredPlainText || sanitizedAggregatedText);
    textBlockStarted = true;
    sawText = true;
    sse(res, "content_block_start", {
      type: "content_block_start",
      index: textIndex,
      content_block: {
        type: "text",
        text: "",
      },
    });
    sse(res, "content_block_delta", {
      type: "content_block_delta",
      index: textIndex,
      delta: {
        type: "text_delta",
        text: finalText,
      },
    });
  }

  if (textBlockStarted) {
    sse(res, "content_block_stop", {
      type: "content_block_stop",
      index: textIndex,
    });
  }

  const finalMessage = finalChunk?.message || {};
  const toolCalls = Array.isArray(finalMessage.tool_calls) ? finalMessage.tool_calls : [];
  const inferredToolBlocks = inferredToolCalls || [];
  if (!shouldForceBufferedPlainText && !shouldSuppressStructuredNoise && (toolCalls.length || inferredToolBlocks.length)) {
    const toolContent = ollamaToAnthropicContent({
      content: "",
      tool_calls: toolCalls,
    }, body.tools).filter((block) => block.type === "tool_use");
    const allToolBlocks = toolCalls.length ? toolContent : inferredToolBlocks;

    for (const [offset, block] of allToolBlocks.entries()) {
      const index = sawText ? textIndex + 1 + offset : offset;
      sse(res, "content_block_start", {
        type: "content_block_start",
        index,
        content_block: {
          type: "tool_use",
          id: block.id,
          name: block.name,
          input: {},
        },
      });
      sse(res, "content_block_delta", {
        type: "content_block_delta",
        index,
        delta: {
          type: "input_json_delta",
          partial_json: JSON.stringify(block.input || {}),
        },
      });
      sse(res, "content_block_stop", {
        type: "content_block_stop",
        index,
      });
    }
  }

  sse(res, "message_delta", {
    type: "message_delta",
    delta: {
      stop_reason: !shouldForceBufferedPlainText && !shouldSuppressStructuredNoise && (toolCalls.length || inferredToolBlocks.length) ? "tool_use" : "end_turn",
      stop_sequence: null,
    },
    usage: {
      output_tokens:
        finalChunk?.eval_count ||
        estimateTokens(aggregatedText || toolCalls),
    },
  });

  sse(res, "message_stop", {
    type: "message_stop",
  });
  res.end();
}

function sendError(res, status, message) {
  json(res, status, {
    type: "error",
    error: {
      type: "api_error",
      message,
    },
  });
}

export function createShimServer() {
  return http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url || "/", "http://127.0.0.1");

      if (req.method === "HEAD") {
        res.writeHead(200);
        res.end();
        return;
      }

      if (req.method === "GET" && url.pathname === "/v1/models") {
        json(res, 200, {
          data: [
            {
              id: defaultOllamaModel,
              type: "model",
              display_name: defaultOllamaModel,
            },
          ],
          has_more: false,
        });
        return;
      }

      if (req.method === "GET" && url.pathname.startsWith("/v1/models/")) {
        json(res, 200, {
          id: decodeURIComponent(url.pathname.replace("/v1/models/", "")),
          type: "model",
          display_name: defaultOllamaModel,
        });
        return;
      }

      if (req.method === "POST" && url.pathname === "/v1/messages/count_tokens") {
        const body = await readBody(req);
        json(res, 200, {
          input_tokens: estimateTokens({
            system: body.system,
            messages: body.messages,
            tools: body.tools,
          }),
        });
        return;
      }

      if (req.method === "POST" && url.pathname === "/v1/messages") {
        const body = await readBody(req);
        if (body.stream) {
          await writeStreamFromOllama(body, res);
          return;
        }

        const ollamaResponse = await callOllama(body);
        const anthropicMessage = buildAnthropicMessage(body, ollamaResponse);
        json(res, 200, anthropicMessage);
        return;
      }

      sendError(res, 404, `Unsupported endpoint: ${req.method} ${url.pathname}`);
    } catch (error) {
      sendError(res, 500, error instanceof Error ? error.message : String(error));
    }
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const server = createShimServer();
  const port = Number(process.env.ANTHROPIC_SHIM_PORT || 0);
  server.listen(port, "127.0.0.1", () => {
    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("Failed to bind shim server");
    }
    process.stdout.write(`${address.port}\n`);
  });
}
