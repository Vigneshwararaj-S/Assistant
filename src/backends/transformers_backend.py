import json
import threading

from transformers import TextIteratorStreamer

from model import Model
from backends.base import ModelBackend

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


def _extract_json_objects(text: str) -> list[str]:
    """Find top-level, brace-balanced {...} substrings in `text`. Unlike a
    regex like r"\\{[^{}]*\\}", this correctly handles nested objects (e.g.
    {"name": ..., "arguments": {"path": "src"}}) as a single candidate."""
    candidates = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            depth, start, j = 0, i, i
            while j < n:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start:j + 1])
                        break
                j += 1
            i = j + 1
        else:
            i += 1
    return candidates


def _overlap_len(text: str, pattern: str) -> int:
    """Length of the longest suffix of `text` that is a prefix of `pattern`."""
    for length in range(min(len(text), len(pattern) - 1), 0, -1):
        if text.endswith(pattern[:length]):
            return length
    return 0


class TransformersBackend(ModelBackend):
    """Loads a model directly into this process via Hugging Face
    transformers + bitsandbytes 4-bit quantization. Supports models with no
    native tool-calling template (e.g. Phi-3.5) via a hand-written prompt
    and raw-text <tool_call> parsing, as well as models with a native
    tools-aware chat template (e.g. Qwen2.5)."""

    def __init__(self, model_name: str, max_new_tokens: int = 256, temperature: float = 0.7):
        self.model = Model(model_name)
        self.tokenizer = self.model.tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def supports_native_tools(self, tool_schemas: list[dict]) -> bool:
        """Probe whether this model's chat template actually renders tool
        definitions, rather than silently ignoring the `tools=` kwarg."""
        probe = [{"role": "user", "content": "x"}]
        try:
            rendered = self.tokenizer.apply_chat_template(
                probe, tools=tool_schemas, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return False
        return any(t["function"]["name"] in rendered for t in tool_schemas)

    def _generate_reply(self, **kwargs):
        streamer = kwargs["streamer"]
        try:
            self.model.model.generate(**kwargs)
        except RuntimeError as e:
            print(f"\n[Generation failed: {e}]")
            streamer.end()

    def generate(self, history: list[dict], tool_schemas: list[dict], native_tools: bool):
        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if native_tools:
            template_kwargs["tools"] = tool_schemas
        template = self.tokenizer.apply_chat_template(history, **template_kwargs)

        input_ids = self.tokenizer(template, return_tensors="pt").input_ids.to(self.model.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            stop_strings=[TOOL_CLOSE],
            tokenizer=self.tokenizer,
        )
        thread = threading.Thread(target=self._generate_reply, kwargs=generation_kwargs)
        thread.start()

        # Scan the whole stream for <tool_call>, not only the first token --
        # the model sometimes writes a sentence before deciding to call a
        # tool. Text before it streams live; the tool call itself is always
        # hidden, and generation stops right at </tool_call>.
        visible_parts = []
        buffer = ""       # trailing text that might still become "<tool_call>"
        hidden = None      # once set, the raw tool-call text (never printed)

        for token in streamer:
            if hidden is not None:
                hidden += token
                continue

            buffer += token
            idx = buffer.find(TOOL_OPEN)
            if idx != -1:
                pre = buffer[:idx]
                if pre:
                    print(pre, end="", flush=True)
                    visible_parts.append(pre)
                hidden = buffer[idx:]
                buffer = ""
                continue

            keep = _overlap_len(buffer, TOOL_OPEN)
            if keep < len(buffer):
                flush = buffer[: len(buffer) - keep]
                print(flush, end="", flush=True)
                visible_parts.append(flush)
                buffer = buffer[len(buffer) - keep:]

        if buffer:
            print(buffer, end="", flush=True)
            visible_parts.append(buffer)

        visible_text = "".join(visible_parts)
        if hidden is None:
            return visible_text, None, None

        # Search for the *last* {"name": ...} object, not the first-to-only-
        # closing-tag span: a confused model can emit garbled or repeated
        # <tool_call> fragments, with the real intended call at the end.
        call = None
        for candidate in reversed(_extract_json_objects(hidden)):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "name" in parsed:
                call = parsed
                break

        if call is None:
            return visible_text, None, (
                "Your last message could not be understood as a tool call. "
                'Respond with a single valid tool call: <tool_call>{"name": "...", "arguments": {...}}</tool_call>'
            )

        return visible_text, {"name": call["name"], "arguments": call.get("arguments", {})}, None

    def format_tool_call_turn(self, tool_call: dict) -> dict:
        text = f"{TOOL_OPEN}{json.dumps(tool_call)}{TOOL_CLOSE}"
        return {"role": "assistant", "content": text}
