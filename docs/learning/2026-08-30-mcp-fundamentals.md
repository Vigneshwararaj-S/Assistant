# MCP Fundamentals

Reference notes from learning the Model Context Protocol (MCP), ahead of building
sub-project 2 (MCP tool-calling) of the local Windows assistant.

## Primary sources

These are the authoritative sources this lesson was built from — go here for
anything this doc simplifies or doesn't cover:

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification/2025-11-25) — the official protocol spec: exact message formats, lifecycle, capability negotiation.
- [modelcontextprotocol/modelcontextprotocol](https://github.com/modelcontextprotocol/modelcontextprotocol) — the spec and documentation source repo on GitHub.
- [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) — the official Python SDK this project will actually use to build the server and client (includes the `FastMCP` decorator API, e.g. `@mcp.tool()`).
- [MCP Blog: The 2026-07-28 Specification](https://blog.modelcontextprotocol.io/posts/2026-07-28/) — notes on the current spec revision, in case protocol details drift from what's summarized below.

## The problem MCP solves

Before MCP, connecting an AI application to a tool required custom, one-off
integration code specific to that exact app-tool pairing. With **M** applications
and **N** tools, you could end up needing **M × N** separate integrations — every
app needing its own bespoke glue code for every tool.

MCP turns this into an **M + N** problem. If every tool exposes itself through the
same standard protocol, and every AI application knows how to speak that same
protocol, any MCP-speaking app can use any MCP-speaking tool with zero custom code
between them.

**Analogy:** MCP is like USB-C. Before USB-C, every device had its own proprietary
connector, requiring a specific cable per device pairing. USB-C became one standard
port any USB-C device can plug into, regardless of manufacturer. MCP is that same
idea for AI applications and tools.

## The three roles: Host, Client, Server

- **Host** — the actual application the human uses. In this project, that's
  `chat.py` — it owns the conversation with the user and talks to the LLM.
- **Client** — a component *inside* the host that manages one connection to one
  specific server. Always a 1-to-1 relationship: a host talking to three servers
  runs three separate clients internally.
- **Server** — a separate program exposing capabilities over the protocol. A server
  has no knowledge of what LLM (if any) is involved — that's the host's concern
  entirely. The server just exposes functionality in a standard way.

In this project: `chat.py` is the **Host**. A small client module inside it is the
**Client**. A separate small program exposing `get_current_datetime` is the
**Server**.

## What a server can expose: three primitives

1. **Tools** — functions the model can actively call to do something or fetch
   something, with potential side effects. Like an API endpoint the model can
   trigger. `get_current_datetime` is a tool. **This project only uses this
   primitive.**
2. **Resources** — passive data the host can read and hand to the model as context,
   without the model actively "calling" anything. More like a file the host
   attaches than an action the model takes.
3. **Prompts** — reusable prompt templates a server can provide.

## The transport layer: JSON-RPC over a channel

MCP messages are formatted as **JSON-RPC 2.0** — a simple standard for saying "call
this named method, with these arguments, tagged with this ID so the response can be
matched back to the request," as a small JSON object. This message *format* is
separate from the *transport* — the channel those messages actually travel over.

- **stdio** — used when the server runs locally as a child process of the client.
  The client launches the server program and communicates by writing JSON-RPC
  messages to its stdin and reading responses from its stdout. The natural choice
  for local, personal-use tools. This project uses stdio.
- **SSE / Streamable HTTP** — used when the server lives on a different machine,
  reachable over a network. Relevant for a future hosted/sellable MCP server, not
  needed here.

## The client-server lifecycle

1. **Connect** — the client launches (or connects to) the server.
2. **Initialize handshake** — client and server exchange a small "here's what I
   support" message so both sides agree on protocol version and capabilities.
3. **Discover** — the client calls `list_tools()`, and the server responds with
   every tool it offers: name, description, and an argument schema. The client
   never hardcodes "the server has tool X" — it asks the server at runtime.
4. **Invoke** — when a tool should be used, the client calls
   `call_tool(name, arguments)` and gets a result back from the server.

## Where the LLM actually fits in

**MCP itself has nothing to do with the LLM.** The protocol only defines how a
Client talks to a Server — no LLM appears anywhere in that exchange. The LLM enters
the picture only because the **Host** application chooses to use one. The host is
responsible for:

- taking the tool list the client discovered and describing it to the model somehow
  (for a model without native tool-calling support, that means writing it into the
  prompt in plain text — see below),
- watching the model's output to detect "the model wants to call a tool,"
- and, when that happens, using the client to invoke the tool and feeding the
  result back to the model.

That bridging logic is **not** part of MCP — it's application code sitting on top
of MCP.

## "Tool calling" without a model trained for it

An LLM only ever does one thing: given text, predict the next chunk of text. There
is no special hidden mode for "tool calling" baked into a model's weights. A model
described as having "native tool-calling support" simply saw many training examples
of a specific structured format (e.g. a particular JSON shape) for requesting a
tool call, so it reproduces that format reliably and stops right after.

A model without that specific training — like **microsoft/Phi-3.5-mini-instruct**,
which has no official tool-calling chat template — can still be taught the same
trick through instructions in the prompt instead of baked into its weights, relying
on its general instruction-following ability: tell it, "if you need today's date,
output exactly `<tool_call>{"name": "get_current_datetime", "arguments": {}}</tool_call>`."
It's less reliable than a model specifically fine-tuned for this, but the mechanism
is the same.

## What actually makes it an "agent"

"Agent" is not a property of the model — it's a property of the loop built around
it. The chat loop from sub-project 1 was: prompt in → model generates → print the
reply. An agent loop adds one decision point:

1. Prompt in, including the list of available tools.
2. Model generates a response.
3. The host inspects that response:
   - **Looks like a normal answer** → show it to the user, done.
   - **Looks like a tool-call request** → don't show the raw text. Parse out the
     tool name and arguments, actually run it via the MCP client, get a real
     result, feed that result into the conversation as new context, then generate
     *again* so the model can produce a real final answer using real data.

That loop — generate, check if it wants to act, act if needed, generate again — is
the entire mechanical definition of an agent. Commercial models (GPT-4, Claude) are
more *reliable* at step 3 due to specialized training, and their APIs often
formalize it into a structured field instead of raw text pattern-matching — but
conceptually it's the same loop, even with a small, non-tool-trained local model.
