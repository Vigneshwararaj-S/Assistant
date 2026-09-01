# MCP Tool-Calling — Design Spec

## Project Context

This is sub-project #2 of the larger "personal Windows assistant" project (see
sub-project #1's spec for the full series context). Sub-project #1 delivered a
working local chat loop (`model.py`, `chat.py`, `main.py`) with streaming,
multi-turn history, error handling, and CLI configuration (`--model_name`,
`--max_new_tokens`, `--temperature`).

This sub-project adds the ability for the model to take real actions — via MCP
(Model Context Protocol) — rather than only producing text. Scope is deliberately
narrow: one MCP server, one trivial tool, proving the full mechanism works
end-to-end before sub-project #3 expands to broader system-control tools.

Background reading: [MCP Fundamentals](../../learning/2026-08-30-mcp-fundamentals.md)
covers the protocol itself (Host/Client/Server roles, transports, primitives) —
this spec assumes that context and focuses on this project's specific design.

## Goal

Prove a real, working agent loop: the model can recognize when it needs external
information it doesn't have (the current date/time), request a tool call in a
parseable format, have that tool actually executed via a real MCP server/client
pair, and produce a correct final answer using the result — all while preserving
the live streaming UX for ordinary (non-tool) conversation turns.

## Constraints

- Same hardware constraint as sub-project #1: RTX 3050 Laptop GPU, 4096 MiB VRAM.
- Must work with the existing default model, **microsoft/Phi-3.5-mini-instruct**,
  which has no official native tool-calling chat template support — tool-calling
  must be achieved via prompt-engineering (describing tools and the required
  response format directly in the prompt), not a built-in structured-output API.
- Must remain model-agnostic: **Qwen/Qwen2.5-3B-Instruct** (already supported via
  the existing `--model_name` flag) has *native* Hermes-style tool-calling using
  the same `<tool_call>{...}</tool_call>` format independently chosen here. No
  architecture should depend on which of the two is used.
- Must use a real, standard MCP server/client (official `mcp` Python SDK, stdio
  transport) — not an in-process function-calling shortcut — since the explicit
  goal of this sub-project is genuinely learning MCP, not just achieving tool use
  by any means.

## Architecture

Three components, each a separate concern:

- **MCP Server** (`mcp_server/tools_server.py`) — a standalone script using
  `FastMCP` from the official SDK, exposing one tool via `@mcp.tool()`. Has no
  knowledge of the LLM or the chat loop — purely a protocol-standard tool
  provider, in principle reusable by any MCP client.
- **MCP Client** (`src/tool_client.py`) — launches the server as a stdio subprocess,
  performs the MCP initialize handshake, exposes `list_tools()` (returns name,
  description, argument schema for each discovered tool) and
  `call_tool(name, arguments)` (invokes a tool, returns its result). Pure protocol
  plumbing — identical regardless of which LLM `chat.py` uses.
- **The Bridge** (inside `chat.py`) — the part specific to this project: injects
  the client's discovered tool list into the prompt, detects when the model's
  output is a tool-call request vs. a normal answer, invokes the client when
  needed, and feeds the result back for a final generation pass.

## Data Flow

1. At `Chat` startup, the MCP client connects to the server and discovers
   available tools via `list_tools()`.
2. Before each generation, the system prompt is built to include a description of
   every discovered tool (name, description, arguments) plus a fixed instruction:
   respond with exactly `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
   if a tool is needed, otherwise answer normally.
3. Generation starts, streamed via `TextIteratorStreamer` as in sub-project #1.
4. **Streaming-decision logic ("Option 2"):** the first ~12 characters of the
   stream are held back (not printed) just long enough to check whether they
   match the `<tool_call>` prefix.
   - If they **don't** match: flush the held text and everything after it straight
     to the terminal, live, exactly as in sub-project #1. Zero UX cost for
     ordinary conversation.
   - If they **do** match: keep suppressing terminal output, keep accumulating
     text until the matching `</tool_call>` is seen, marking this generation as a
     tool-call attempt rather than a user-facing answer.
5. If it was a tool-call attempt: parse the JSON between the tags, call
   `call_tool(name, arguments)` via the MCP client, and append the result to
   history as a `"user"`-role message with a distinguishing prefix (e.g.
   `"[Tool result for get_current_datetime]: ..."`) — chosen over a dedicated
   `"tool"` role since Phi-3.5-mini's chat template doesn't recognize one. Then
   generate again — this second pass **does** stream live, since it's expected to
   produce the real final answer.
6. If it was a normal answer: nothing further needed, it already streamed live in
   step 4.
7. The final assistant reply (never the raw `<tool_call>` text) is what gets
   appended to `history` as the assistant's turn.

## The First Tool

`get_current_datetime` — takes no arguments, returns the current date and time as
a string. Deliberately trivial: this sub-project is about proving the mechanism,
not the tool. The model cannot produce a correct current time on its own, so a
correct final answer is unambiguous proof the round trip actually worked, not a
lucky guess.

## Error Handling

- **Malformed tool-call JSON** (model emits `<tool_call>` but invalid JSON inside):
  catch the parse error, don't crash — treat as a failed tool attempt and surface
  a plain message, not a raw traceback.
- **Tool execution failure** (the MCP server's `call_tool` raises): catch it, feed
  the error back to the model as the "tool result" (e.g. `"[Tool error]: ..."`) so
  the model can inform the user itself, rather than the turn silently dying.
- **MCP server fails to start / connection fails**: fail loudly at `Chat`
  initialization with a clear message — without the server, tool-calling can't
  work at all for this sub-project, so this should not be silently swallowed.

Explicitly out of scope for v1: automatic retry on malformed tool-call format.
Worth a future enhancement, but adds real complexity (retry limits, context cost)
not needed to prove the core mechanism.

## Testing

Manual verification, same style as sub-project #1:

- MCP server starts; the client's `list_tools()` correctly reports
  `get_current_datetime`.
- A prompt that needs the time (e.g. "what time is it?") triggers a tool call,
  executes it, and the model's final answer reflects the real result.
- A prompt that doesn't need a tool streams live with no pause or false-positive
  tool detection.
- Malformed tool-call JSON and tool execution failure both produce a graceful
  message, not a crash.
- Model-agnostic claim: run once with `--model_name microsoft/Phi-3.5-mini-instruct`
  and once with `--model_name Qwen/Qwen2.5-3B-Instruct`; both complete a
  tool-calling round trip with no code changes.

## Success Criteria

- A real MCP server (official SDK, stdio transport) exposing `get_current_datetime`
  is running and independently connectable.
- A real MCP client in `chat.py` discovers and invokes that tool dynamically (no
  hardcoded tool metadata in the client).
- A time-dependent question gets a correct, tool-backed answer.
- Ordinary conversation retains full live streaming, unaffected by the new logic.
- Both configured models (Phi-3.5-mini via prompting, Qwen2.5-3B via native
  support) can complete the round trip.

## Out of Scope (deferred to future sub-projects)

- Broad system-control tools (files, apps) — sub-project #3.
- Multi-hop tool use (a single turn calling more than one tool in sequence).
- Automatic retry on malformed tool-call output.
- Multi-model orchestration / on-demand specialist model loading — future
  sub-project, and explicitly premature until multiple distinct tools exist to
  route between.
- A dedicated `"tool"` chat-template role — using a prefixed `"user"` message
  instead, since Phi-3.5-mini's template doesn't support one natively.
