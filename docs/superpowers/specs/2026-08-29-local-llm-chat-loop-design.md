# Local LLM Chat Loop — Design Spec

## Project Context

This is sub-project #1 of a larger "personal Windows assistant" project, being built
and documented as a LinkedIn learning series. The full vision is a local LLM/SLM,
orchestrated with MCP tools, that can act as a Windows assistant: browsing the web,
installing software, scanning for malware, controlling media playback, and executing
commands.

That full scope is too large for a single spec. It has been decomposed into
independent sub-projects, each shippable and postable on its own:

1. **Local LLM chat loop** (this spec) — get a Hugging Face SLM running locally and
   talking back through a terminal REPL. No tools yet.
2. MCP tool-calling with a first real tool (future)
3. System-control capabilities — files, apps (future)
4. Web access (future)
5. Malware scanning integration (future)
6. Media control (e.g. YouTube playback) (future)

Each future sub-project gets its own spec → plan → implementation cycle once this one
ships.

## Goal

Run a small instruction-tuned LLM entirely on the user's laptop GPU and have a
working, streaming terminal chat with it. This proves local inference works, is fast
enough to be usable, and gives the user hands-on familiarity with the Hugging Face
`transformers` stack (tokenizers, chat templates, quantized generation) — the raw
material for the first LinkedIn post in the series.

## Constraints

- Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB VRAM.
- Model must run 4-bit quantized to comfortably fit VRAM alongside KV cache.
- Model must be ungated on the Hugging Face Hub (no license click-through / HF token
  required) to keep first-run setup friction low.
- Windows 11, PowerShell-primary environment.

## Model Choice

**microsoft/Phi-3.5-mini-instruct** — 3.8B parameters, MIT license, ungated,
4-bit footprint ≈ 2.2GB. Chosen by the user (curiosity about the model, not a hard
technical requirement over alternatives like Qwen2.5-3B-Instruct, which was also
evaluated and would be an equally valid choice).

## Architecture

A small Python CLI application with three modules:

- `model.py` — loads the tokenizer and model from the HF Hub with a
  `BitsAndBytesConfig(load_in_4bit=True, ...)`, places the model on the GPU
  (`device_map="auto"`).
- `chat.py` — the REPL loop: keeps conversation history as a list of
  `{role, content}` messages, applies the model's chat template via
  `tokenizer.apply_chat_template()`, and streams generation back to the terminal
  token-by-token using `TextIteratorStreamer`.
- `main.py` — entry point. CLI flags for model id (defaulted to Phi-3.5-mini-instruct),
  `max_new_tokens`, and `temperature`.

## Data Flow

1. User types a message at the prompt.
2. Message is appended to in-memory conversation history.
3. Full history is rendered through `apply_chat_template()` into the model's expected
   prompt format.
4. Prompt is tokenized and passed to `model.generate()` running in a background
   thread, wired to a `TextIteratorStreamer`.
5. Tokens are printed to stdout as they arrive.
6. The complete assistant reply is appended back into history as an `assistant` turn.
7. Loop repeats.

REPL commands: `/reset` clears history, `/exit` quits.

## Error Handling

- First run downloads ~2GB of weights from the Hub — print a one-time
  "downloading model, this happens once" notice before the download starts.
- CUDA out-of-memory during generation: catch the exception, print a plain-English
  suggestion (lower `max_new_tokens`, or note the model may need swapping for a
  smaller one). Do not crash with a raw stack trace.
- `Ctrl+C` during generation or at the prompt exits cleanly.

## Testing

This is a hands-on demo script, not a library, so no formal automated test suite for
v1. Verification is manual:

- Model loads successfully and reports which device it's running on.
- A prompt produces a coherent, streamed response within a few seconds.
- Conversation history persists correctly across multiple turns (the model can refer
  back to earlier turns).
- `/reset` and `/exit` behave as expected.

## Success Criteria

- Model loads and runs entirely on the RTX 3050 GPU (no CPU fallback needed).
- A short prompt gets a fully streamed response in a few seconds.
- Multi-turn conversation history works correctly.
- Working code + a captured terminal session/log suitable as the basis for the first
  LinkedIn post in the series.

## Out of Scope (deferred to future sub-projects)

- Any tool-calling / MCP integration.
- Any system, file, or web access.
- Any GUI — terminal only for this phase.
- Persisting conversation history across process restarts.
