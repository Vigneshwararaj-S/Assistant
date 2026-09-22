# Assistant

A local, personal Windows AI assistant — a small language model running
entirely on your own machine, given real tools (via MCP) to access files,
launch/close apps, and browse the web.

## Setup

**1. Python environment**
```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```
`torch`/`torchvision`/`torchaudio` need a CUDA-matched build — see the
comment at the top of `requirements.txt` for the exact install command.

**2. Model backend** — pick one (or both):
- **Ollama (default)** — install [Ollama](https://ollama.com), then pull the
  default model: `ollama pull qwen2.5:3b`
- **transformers** — no extra setup; models download automatically on first
  run. Use `--backend transformers --model_name <hf-repo-id>`.

**3. Web search (optional)** — `web_search` needs a local
[SearXNG](https://docs.searxng.org/) instance:
```
mkdir .searxng\config .searxng\data
docker run --name searxng -d -p 8080:8080 ^
    -v "%cd%\.searxng\config:/etc/searxng" ^
    -v "%cd%\.searxng\data:/var/cache/searxng" ^
    docker.io/searxng/searxng:latest
```
Then edit `.searxng\config\settings.yml`, add under `search:`:
```yaml
search:
  formats:
    - html
    - json
```
and `docker restart searxng`. (JSON output is off by default, even for a
self-hosted instance.) `fetch_url` and the other tools work fine without
this — only `web_search` needs it running.

> **Note (Windows + Git Bash):** if setting this up from Git Bash rather
> than PowerShell/cmd, its automatic path rewriting can corrupt the `-v`
> volume flags above (a `/etc/searxng` destination silently becomes
> `/Program Files/Git/etc/searxng`). Run the `docker run` command from
> PowerShell or cmd instead.

## Running

```
venv\Scripts\python src\main.py
```

Flags: `--backend {transformers,ollama}`, `--model_name`, `--max_new_tokens`,
`--temperature`, `--files_root` (sandbox root for file tools, default `.`).

In the chat: `/reset` clears history, `/exit` quits. `write_file`,
`delete_file`, `launch_app`, and `close_app` ask for confirmation before
running; everything else runs automatically.

## Architecture

- `src/main.py` — entry point; launches the MCP tool servers and starts the chat loop.
- `src/chat.py` — the agent loop: builds prompts, detects and executes tool calls, handles multi-hop tool chains.
- `src/backends/` — pluggable model runtimes (`transformers_backend.py`, `ollama_backend.py`), a common interface `chat.py` calls without caring which one is in use.
- `src/tool_client.py` — a real MCP client, connects to the tool servers over stdio.
- `src/mcp_server/` — the MCP servers themselves: `tools_server.py` (time), `filesystem_server.py` (read/write, sandboxed), `apps_server.py` (launch/close, allow-listed), `web_server.py` (fetch/search).

Full design history and decisions: `docs/superpowers/`.
