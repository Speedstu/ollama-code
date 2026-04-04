# Mirror Gaps

This repository contains a large mirrored Claude Code source tree, but it is not a complete, build-ready upstream checkout.

Known issues in this mirror:

- No `package.json`, lockfile, or TypeScript build configuration were present.
- Some imports point to files that are absent from the mirror, so a straight `bun build` or `npm install && npm run build` path is not currently possible.
- Several modules still target Anthropic-specific services and private infrastructure.

Because of that, this repo now includes a standalone `ollama-code` runtime in [`bin/ollama-code.mjs`](/home/uwucode/1synapse/Synapse-main/ollama-code/bin/ollama-code.mjs) that uses Ollama directly while keeping the mirrored source tree available to the model through local tools.
