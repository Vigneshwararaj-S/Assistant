# MCP Tool-Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Note for this project specifically:** this project runs in guide mode (see
> project memory `feedback_teach_dont_just_execute` / `feedback_execution_preference`)
> — the user writes the code themselves with Claude teaching and reviewing each
> step, rather than a subagent or Claude executing tasks autonomously. If resuming
> this plan in guide mode, walk through each task's steps directly with the user
> instead of dispatching subagents.

**Goal:** Give the local chat loop the ability to call a real MCP tool
(`get_current_datetime`) via a genuine MCP server/client pair, proving the full
agent loop works end-to-end while preserving live streaming for ordinary
conversation.

**Architecture:** A standalone MCP server (official SDK, `MCPServer` +
`@mcp.tool()`, stdio transport) exposes one tool. A persistent MCP client, run on
a dedicated background thread with its own asyncio event loop (bridged into the
existing synchronous `chat.py` via `run_coroutine_threadsafe`), discovers and
invokes it. `chat.py` injects the discovered tool list into the prompt, and uses
a lookahead buffer on the token stream to detect a `<tool_call>{...}</tool_call>`
request without sacrificing live streaming for normal replies.

**Tech Stack:** Official `mcp` Python SDK (v2, `MCPServer`/`Client`/
`StdioServerParameters`), Python `asyncio` + `threading`, existing `transformers`/
`TextIteratorStreamer` stack from sub-project 1.

**Spec:** `docs/superpowers/specs/2026-08-30-mcp-tool-calling-design.md`

## Global Constraints

- Default model stays **microsoft/Phi-3.5-mini-instruct**; the tool-calling
  bridge must also work unmodified with **Qwen/Qwen2.5-3B-Instruct** via the
  existing `--model_name` flag (per spec's model-agnostic requirement).
- Tool-call request format is fixed: `<tool_call>{"name": "...", "arguments":
  {...}}</tool_call>` — exact tag text, JSON body.
- Tool results are fed back as a `"user"`-role message with a
  `"[Tool result for <name>]: "` prefix — no `"tool"` role (Phi-3.5-mini's
  template doesn't support one).
- No automated test suite exists in this project (confirmed: no `tests/` folder,
  no `pytest` usage anywhere in `src/`) — sub-project 1 established manual
  verification as the project's testing convention; this plan follows the same
  pattern rather than introducing `pytest` unilaterally.
- MCP SDK API verified directly against `modelcontextprotocol/python-sdk`
  source files on 2026-08-30 (this is the current **v2** SDK — a major rework;
  older MCP tutorials/blog posts using `FastMCP`/`stdio_client`/`ClientSession`
  reflect the pre-v2 API and do not apply here).

---

## File Structure

- Create: `mcp_server/tools_server.py` — the MCP server, one tool.
- Create: `src/tool_client.py` — the MCP client wrapper (`ToolClient` class),
  persistent connection via a background event-loop thread.
- Modify: `src/chat.py` — inject tool descriptions into the prompt, add the
  lookahead streaming-detection logic, parse and handle tool calls, error
  handling.
- Modify: `src/main.py` — construct and close the `ToolClient`, pass it to `Chat`.

---

### Task 1: MCP Server with `get_current_datetime`

**Files:**
- Create: `mcp_server/tools_server.py`

**Interfaces:**
- Produces: a runnable script that, launched as a subprocess, serves one MCP
  tool named `get_current_datetime` (no arguments, returns an ISO-format
  datetime string) over stdio.

- [ ] **Step 1: Write the server script**

```python
from datetime import datetime

from mcp.server import MCPServer

mcp = MCPServer("AssistantTools")


@mcp.tool()
def get_current_datetime() -> str:
    """Get the current local date and time, ISO 8601 format."""
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 2: Verify the tool function works in isolation**

Run (from the project root, with the venv active):
```
python -c "from mcp_server.tools_server import get_current_datetime; print(get_current_datetime())"
```
Expected: prints something like `2026-08-30T21:14:03` — proves the underlying
function is correct before involving the MCP protocol layer at all. (Note: this
calls the plain Python function directly, bypassing the `@mcp.tool()` wrapper —
full protocol-level verification happens in Task 2, once a client exists that can
actually speak MCP to this server.)

- [ ] **Step 3: Commit**

```bash
git add mcp_server/tools_server.py
git commit -m "feat: add MCP server exposing get_current_datetime tool"
```

---

### Task 2: MCP Client (`ToolClient`)

**Files:**
- Create: `src/tool_client.py`

**Interfaces:**
- Consumes: `mcp_server/tools_server.py` (Task 1), launched as a subprocess.
- Produces: `ToolClient` class with:
  - `__init__(self, command: str, args: list[str])`
  - `list_tools(self) -> list` — each item has `.name`, `.description`
  - `call_tool(self, name: str, arguments: dict) -> Any` — the tool's
    `structured_content` result
  - `close(self)`

- [ ] **Step 1: Write `ToolClient`**

```python
import asyncio
import threading

from mcp import Client, StdioServerParameters


class ToolClient:
    def __init__(self, command: str, args: list[str]):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        server = StdioServerParameters(command=command, args=args)
        self._client_ctx = Client(server)
        self._client = self._run_coro(self._client_ctx.__aenter__())

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def list_tools(self):
        result = self._run_coro(self._client.list_tools())
        return result.tools

    def call_tool(self, name: str, arguments: dict):
        result = self._run_coro(self._client.call_tool(name, arguments))
        return result.structured_content

    def close(self):
        self._run_coro(self._client_ctx.__aexit__(None, None, None))
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
```

- [ ] **Step 2: Verify the full protocol round trip**

Write a throwaway check script (not part of the permanent codebase — delete
after running), e.g. `scratch_verify_client.py` in the project root:
```python
import sys

from src.tool_client import ToolClient

client = ToolClient(command=sys.executable, args=["mcp_server/tools_server.py"])
tools = client.list_tools()
print("Discovered tools:", [t.name for t in tools])
assert any(t.name == "get_current_datetime" for t in tools), "tool not discovered"

result = client.call_tool("get_current_datetime", {})
print("Tool result:", result)

client.close()
print("OK")
```
Run: `python scratch_verify_client.py`
Expected: prints the discovered tool list including `get_current_datetime`, then
the actual current datetime string, then `OK`, with no exceptions. This is the
first real proof the MCP server and client can talk to each other at all.

Delete `scratch_verify_client.py` once this passes — it was only to prove the
round trip works before wiring it into `chat.py`.

- [ ] **Step 3: Commit**

```bash
git add src/tool_client.py
git commit -m "feat: add MCP client (ToolClient) with persistent stdio connection"
```

---

### Task 3: Wire tool-calling into `chat.py`

**Files:**
- Modify: `src/chat.py`

**Interfaces:**
- Consumes: `ToolClient` (Task 2) — `list_tools()`, `call_tool(name, arguments)`.
- Produces: `Chat.__init__` now accepts a `tool_client: ToolClient` parameter;
  `Chat.chat()` behavior unchanged from the outside (still takes a `prompt: str`,
  still streams the final answer to the terminal) but now transparently handles
  tool calls internally.

- [ ] **Step 1: Accept the `ToolClient` and build the tool-aware system prompt**

In `Chat.__init__`, store the client and build a system-prompt fragment
describing the discovered tools:
```python
def __init__(self, model_name: str, tool_client, max_new_tokens: int = 256, temperature: float = 0.7):
    self.model = Model(model_name)
    self.tokenizer = self.model.tokenizer
    self.tool_client = tool_client
    self.max_new_tokens = max_new_tokens
    self.temperature = temperature

    tools = self.tool_client.list_tools()
    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}" for t in tools
    )
    system_prompt = (
        "You have access to the following tools:\n"
        f"{tool_descriptions}\n\n"
        "If you need to use a tool, respond with EXACTLY this format and "
        "nothing else:\n"
        '<tool_call>{"name": "<tool_name>", "arguments": {}}</tool_call>\n'
        "Otherwise, answer normally."
    )
    self.history = [{"role": "system", "content": system_prompt}]
```

- [ ] **Step 2: Extract generation into a reusable helper**

Refactor the existing generate-and-stream logic (thread + streamer setup) out of
`chat()` into a private helper that returns the raw generated text, so it can be
called twice (once for the possible tool-call, once for the final answer after a
tool result). Since the lookahead buffering needs the raw stream, this helper
takes a `stream_live: bool` flag:

```python
def _generate(self, stream_live: bool) -> str:
    template = self.tokenizer.apply_chat_template(
        self.history, tokenize=False, add_generation_prompt=True
    )
    input_ids = self.tokenizer(template, return_tensors="pt").input_ids.to(self.model.model.device)
    streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=self.max_new_tokens,
        temperature=self.temperature,
        do_sample=True,
    )
    thread = threading.Thread(target=self._generate_reply, kwargs=generation_kwargs)
    thread.start()

    TOOL_PREFIX = "<tool_call>"
    buffer = ""
    decided = False
    is_tool_call = False
    full_text = ""

    for new_token in streamer:
        full_text += new_token
        if not decided:
            buffer += new_token
            if len(buffer) >= len(TOOL_PREFIX):
                is_tool_call = buffer.startswith(TOOL_PREFIX)
                decided = True
                if stream_live and not is_tool_call:
                    print(buffer, end="", flush=True)
            continue
        if stream_live and not is_tool_call:
            print(new_token, end="", flush=True)

    return full_text
```

- [ ] **Step 3: Rewrite `chat()` to use the tool-call round trip**

```python
def chat(self, prompt: str):
    if prompt.strip().lower() == "/exit":
        print("Exiting chat...")
        return

    if prompt.strip().lower() == "/reset":
        self.history = self.history[:1]  # keep the system prompt
        print("Chat history reset.")
        return

    self.history.append({"role": "user", "content": prompt})

    print("\nAssistant: ", end="")
    reply = self._generate(stream_live=True)

    if reply.strip().startswith("<tool_call>"):
        tool_result = self._handle_tool_call(reply)
        self.history.append({"role": "user", "content": tool_result})
        print("\nAssistant: ", end="")
        reply = self._generate(stream_live=True)

    self.history.append({"role": "assistant", "content": reply})
```

- [ ] **Step 4: Implement `_handle_tool_call` with error handling**

```python
import json

def _handle_tool_call(self, raw_reply: str) -> str:
    try:
        json_text = raw_reply.strip().removeprefix("<tool_call>").removesuffix("</tool_call>")
        call = json.loads(json_text)
        name = call["name"]
        arguments = call.get("arguments", {})
    except (json.JSONDecodeError, KeyError) as e:
        return f"[Tool call could not be parsed: {e}]"

    try:
        result = self.tool_client.call_tool(name, arguments)
    except Exception as e:
        return f"[Tool error for {name}]: {e}"

    return f"[Tool result for {name}]: {result}"
```

- [ ] **Step 5: Verify manually**

Run `python main.py` and:
1. Ask "what time is it?" — expect a pause (tool call happening silently), then
   a streamed final answer containing the real current time.
2. Ask an ordinary question, e.g. "what's 2+2?" — expect it to stream live
   immediately with no pause, same as before this change.
3. Confirm `/reset` and `/exit` still work.

- [ ] **Step 6: Commit**

```bash
git add src/chat.py
git commit -m "feat: add MCP tool-calling round trip with streaming-preserving detection"
```

---

### Task 4: Wire `ToolClient` into `main.py`, with startup error handling

**Files:**
- Modify: `src/main.py`

**Interfaces:**
- Consumes: `ToolClient` (Task 2), `Chat.__init__` now requiring `tool_client`
  (Task 3).

- [ ] **Step 1: Construct the `ToolClient` before `Chat`, with a clear failure message**

```python
import sys

from tool_client import ToolClient

def main():
    args = parser.parse_args()

    try:
        tool_client = ToolClient(
            command=sys.executable,
            args=["mcp_server/tools_server.py"],
        )
    except Exception as e:
        print(f"Failed to start the MCP tool server: {e}")
        return

    chat_app = Chat(args.model_name, tool_client, args.max_new_tokens, args.temperature)
    try:
        while True:
            user_input = input("\nUser: ")
            chat_app.chat(user_input)
            if user_input.strip().lower() == "/exit":
                break
    except KeyboardInterrupt:
        print("\nExiting chat...")
    finally:
        tool_client.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the model-agnostic claim from the spec**

Run once with the default model, then once explicitly with Qwen:
```
python main.py
python main.py --model_name Qwen/Qwen2.5-3B-Instruct
```
For each: ask a time-dependent question and confirm a correct, tool-backed
answer comes back. This is the spec's success criterion that the design doesn't
depend on which model is loaded.

- [ ] **Step 3: Commit**

```bash
git add src/main.py
git commit -m "feat: start MCP tool server from main.py with startup error handling"
```

---

## Self-Review

**Spec coverage:**
- Real MCP server/client, stdio transport, official SDK → Tasks 1–2. ✓
- Tool discovery injected into prompt, not hardcoded → Task 3 Step 1. ✓
- Streaming-preserving lookahead detection ("Option 2") → Task 3 Step 2. ✓
- Tool result fed back as prefixed `"user"` message → Task 3 Steps 3–4. ✓
- Malformed JSON / tool execution failure handled gracefully → Task 3 Step 4. ✓
- MCP server startup failure fails loudly → Task 4 Step 1. ✓
- Model-agnostic verification (Phi-3.5-mini + Qwen2.5-3B) → Task 4 Step 2. ✓
- `get_current_datetime` as the first tool → Task 1. ✓

**Placeholder scan:** no TBD/TODO; all steps show complete, runnable code.

**Type consistency:** `ToolClient.call_tool(name: str, arguments: dict)` used
identically in Task 2 (definition) and Task 3 Step 4 (`_handle_tool_call`).
`Chat.__init__(model_name, tool_client, max_new_tokens, temperature)` matches the
positional call in Task 4 Step 1. `_generate(stream_live: bool) -> str` is defined
once in Task 3 Step 2 and called identically (with `stream_live=True`) in both
places in Task 3 Step 3.
