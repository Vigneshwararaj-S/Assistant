import json
import threading
from transformers import TextIteratorStreamer
from model import Model

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
MAX_TOOL_HOPS = 4


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


class Chat:
    def __init__(self, model_name: str, tool_clients, max_new_tokens: int = 256, temperature: float = 0.7):
        self.model = Model(model_name)
        self.tokenizer = self.model.tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._tool_owner = {}
        tools = []
        for tc in tool_clients:
            for t in tc.list_tools():
                self._tool_owner[t.name] = tc
                tools.append(t)
        self.tool_schemas = [self._to_openai_schema(t) for t in tools]
        self.native_tools = self._supports_native_tools()

        if self.native_tools:
            self._base_history = []
        else:
            tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools)
            system_prompt = (
                "You have access to the following tools:\n"
                f"{tool_descriptions}\n\n"
                "If you need to use a tool, respond with EXACTLY this format and "
                "nothing else:\n"
                f'{TOOL_OPEN}{{"name": "<tool_name>", "arguments": {{}}}}{TOOL_CLOSE}\n'
                "You may call a tool more than once in a row if you need another "
                "one after seeing a result. Otherwise, answer normally."
            )
            self._base_history = [{"role": "system", "content": system_prompt}]
        self.history = list(self._base_history)
        self._last_call = None

    @staticmethod
    def _to_openai_schema(tool):
        params = (
            getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", None)
            or {"type": "object", "properties": {}}
        )
        return {
            "type": "function",
            "function": {"name": tool.name, "description": tool.description, "parameters": params},
        }

    def _supports_native_tools(self) -> bool:
        """Probe whether this model's chat template actually renders tool
        definitions, rather than silently ignoring the `tools=` kwarg."""
        probe = [{"role": "user", "content": "x"}]
        try:
            rendered = self.tokenizer.apply_chat_template(
                probe, tools=self.tool_schemas, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return False
        return any(t["function"]["name"] in rendered for t in self.tool_schemas)

    def _generate_reply(self, **kwargs):
        streamer = kwargs["streamer"]
        try:
            self.model.model.generate(**kwargs)
        except RuntimeError as e:
            print(f"\n[Generation failed: {e}]")
            streamer.end()

    def chat(self, prompt: str):
        if prompt.strip().lower() == "/exit":
            print("Exiting chat...")
            return

        if prompt.strip().lower() == "/reset":
            self.history = list(self._base_history)
            print("Chat history reset.")
            return

        self.history.append({"role": "user", "content": prompt})
        print("\nAssistant: ", end="")

        self._last_call = None  # only suppress repeats within this one turn
        visible, tool_call_text = self._generate()
        answer = visible
        hops = 0
        while tool_call_text is not None and hops < MAX_TOOL_HOPS:
            hops += 1
            name, result = self._handle_tool_call(tool_call_text)
            if self.native_tools:
                self.history.append({"role": "assistant", "content": tool_call_text.strip()})
                self.history.append({"role": "tool", "content": str(result)})
            else:
                label = name or "tool"
                self.history.append({
                    "role": "user",
                    "content": (
                        f"[Tool result for {label}]: {result}\n"
                        "(This is real, current data from the tool. Use it to answer the user's last question.)"
                    ),
                })
            visible, tool_call_text = self._generate()
            answer += visible

        if tool_call_text is not None:
            msg = "\n[Stopped after too many tool calls in a row without a final answer.]"
            print(msg, end="", flush=True)
            answer += msg

        self.history.append({"role": "assistant", "content": answer})

    def _handle_tool_call(self, raw_reply: str):
        """Parse and execute a <tool_call> block. Returns (name, result) on
        success, or (name_or_None, error_message) on failure — never raises.

        Searches for the *last* {"name": ...} object in the raw text, not the
        first-to-only-closing-tag span: a confused model can emit garbled or
        repeated <tool_call> fragments, with the real intended call at the end.
        """
        call = None
        for candidate in reversed(_extract_json_objects(raw_reply)):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "name" in parsed:
                call = parsed
                break

        if call is None:
            print("[tool call failed to parse] ", end="", flush=True)
            return None, (
                "Your last message could not be understood as a tool call. "
                'Respond with a single valid tool call: <tool_call>{"name": "...", "arguments": {...}}</tool_call>'
            )
        name = call["name"]
        arguments = call.get("arguments", {})

        key = (name, json.dumps(arguments, sort_keys=True))
        if key == self._last_call:
            print(f"[skipping repeated call to {name}] ", end="", flush=True)
            return name, (
                "You already called this tool with these exact arguments — see the "
                "result above. Do not call it again; answer the user's question now."
            )
        self._last_call = key

        owner = self._tool_owner.get(name)
        if owner is None:
            print(f"[unknown tool: {name}] ", end="", flush=True)
            return name, f"There is no tool named '{name}'. Use one of the tools listed above."

        print(f"[calling tool: {name}] ", end="", flush=True)
        try:
            result = owner.call_tool(name, arguments)
        except Exception as e:
            return name, f"Tool error: {e}"

        if isinstance(result, dict) and set(result) == {"result"}:
            result = result["result"]
        return name, result

    def _generate(self):
        """Generate one model turn, scanning the whole stream for a
        <tool_call> block — not only at the very start, since the model
        sometimes writes a sentence before deciding to call a tool.

        Text before any tool call streams live as it's generated. The tool
        call itself is always hidden, and generation stops right at
        </tool_call>. Returns (visible_text, tool_call_text); tool_call_text
        is None if no tool call appeared anywhere in this generation.
        """
        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if self.native_tools:
            template_kwargs["tools"] = self.tool_schemas
        template = self.tokenizer.apply_chat_template(self.history, **template_kwargs)

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

        return "".join(visible_parts), hidden
