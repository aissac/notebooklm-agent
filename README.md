# 🧠 notebooklm-agent

Zero-token AI agent powered by Google NotebookLM.

No API keys. No external LLMs. NotebookLM's built-in Gemini does all the reasoning.

## Features

- **Zero external tokens** — NotebookLM's Gemini is free
- **Brain-first architecture** — every notebook auto-bootstrapped before use
- **Atomic research pipeline** — `brain_research()` does start→poll→import→wait
- **Persistent memory** — notebooks grow smarter over time
- **Multi-gateway** — Telegram, Discord, CLI, WebSocket
- **Custom ReAct loop** — no smolagents dependency, direct `chat.ask()`
- **Async-native** — no sync/async bridge hacks

## Install

```bash
pip install notebooklm-agent
```

## Quick Start

```bash
# Login to Google (opens browser for OAuth cookies)
nlm-agent login

# Chat with a new notebook (auto-bootstrapped + auto-researched)
nlm-agent run "Explain quantum computing"

# Start Telegram bot
nlm-agent serve --gateway telegram

# Research a topic
nlm-agent research "fusion energy 2025"

# Generate podcast from research
nlm-agent podcast
```

## Architecture

```
User → Gateway (Telegram/CLI/Web)
         ↓
      Agent Core (ReAct loop)
         ↓
      Brain (NotebookLM chat.ask)
         ↓
      notebooklm-py (API client)
```

Every notebook gets bootstrapped with agent instructions before first use.
Every question goes through Gemini via `chat.ask()` — no local model needed.

## License

Apache-2.0