import json
import re
import threading
from transformers import TextIteratorStreamer
from model import Model

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


class Chat:
    def __init__(self, model_name: str, tool_client, max_new_tokens: int = 256, temperature: float = 0.7):
        self.model = Model(model_name)
        self.tokenizer = self.model.tokenizer
        self.tool_client = tool_client
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        tools = self.tool_client.list_tools()
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
                "Otherwise, answer normally."
            )
            self._base_history = [{"role": "system", "content": system_prompt}]
        self.history = list(self._base_history)

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
        reply = self._generate(expect_tool_call=True)

        if reply.strip().startswith(TOOL_OPEN):
            name, result = self._handle_tool_call(reply)
            if self.native_tools:
                self.history.append({"role": "assistant", "content": reply.strip()})
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
            reply = self._strip_tool_calls(self._generate(expect_tool_call=False))

        self.history.append({"role": "assistant", "content": reply})

    def _strip_tool_calls(self, text: str) -> str:
        return re.sub(rf"{TOOL_OPEN}.*?{TOOL_CLOSE}\s*", "", text, flags=re.DOTALL).strip()

    def _handle_tool_call(self, raw_reply: str):
        """Parse and execute a <tool_call> block. Returns (name, result) on
        success, or (name_or_None, error_message) on failure — never raises."""
        match = re.search(rf"{TOOL_OPEN}(.*?)(?:{TOOL_CLOSE}|$)", raw_reply.strip(), re.DOTALL)
        try:
            call = json.loads(match.group(1))
            name = call["name"]
            arguments = call.get("arguments", {})
        except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as e:
            return None, f"Tool call could not be parsed: {e!r}"

        print(f"[calling tool: {name}] ", end="", flush=True)
        try:
            result = self.tool_client.call_tool(name, arguments)
        except Exception as e:
            return name, f"Tool error: {e}"

        if isinstance(result, dict) and set(result) == {"result"}:
            result = result["result"]
        return name, result

    def _generate(self, expect_tool_call: bool) -> str:
        """Generate a reply, streaming it live unless it is a tool call.

        expect_tool_call=True (first pass): a reply starting with <tool_call> is
        held back (never printed) and generation stops at the closing tag.
        expect_tool_call=False (after a tool result): any echoed <tool_call>
        block is swallowed and the real answer streams live.
        Returns the raw generated text.
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
        )
        if expect_tool_call:
            generation_kwargs.update(stop_strings=[TOOL_CLOSE], tokenizer=self.tokenizer)

        thread = threading.Thread(target=self._generate_reply, kwargs=generation_kwargs)
        thread.start()

        state = "deciding"  # deciding -> streaming | tool_call | swallowing
        pending = ""
        full_text = ""

        for token in streamer:
            full_text += token
            if state == "streaming":
                print(token, end="", flush=True)
                continue

            pending += token
            if state == "deciding":
                if pending.startswith(TOOL_OPEN):
                    state = "tool_call" if expect_tool_call else "swallowing"
                elif TOOL_OPEN.startswith(pending):
                    continue  # could still turn into a tool call, keep holding
                else:
                    state = "streaming"
                    print(pending, end="", flush=True)
                    continue

            if state == "swallowing" and TOOL_CLOSE in pending:
                state = "streaming"
                rest = pending.split(TOOL_CLOSE, 1)[1].lstrip()
                print(rest, end="", flush=True)

        if state == "deciding":
            print(pending, end="", flush=True)

        return full_text
