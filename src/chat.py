import json

MAX_TOOL_HOPS = 4
CONFIRM_REQUIRED = {"write_file", "delete_file", "launch_app", "close_app"}
CONFIRM_PREVIEW_CHARS = 300


class Chat:
    def __init__(self, backend, tool_clients):
        self.backend = backend

        self._tool_owner = {}
        tools = []
        for tc in tool_clients:
            for t in tc.list_tools():
                self._tool_owner[t.name] = tc
                tools.append(t)
        self.tool_schemas = [self._to_openai_schema(t) for t in tools]
        self.native_tools = backend.supports_native_tools(self.tool_schemas)

        if self.native_tools:
            self._base_history = []
        else:
            tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools)
            system_prompt = (
                "You have access to the following tools:\n"
                f"{tool_descriptions}\n\n"
                "If you need to use a tool, respond with EXACTLY this format and "
                "nothing else:\n"
                '<tool_call>{"name": "<tool_name>", "arguments": {}}</tool_call>\n'
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
        visible, tool_call, parse_error = self.backend.generate(
            self.history, self.tool_schemas, self.native_tools
        )
        answer = visible
        hops = 0
        while (tool_call is not None or parse_error is not None) and hops < MAX_TOOL_HOPS:
            hops += 1
            if parse_error is not None:
                print("[tool call failed to parse] ", end="", flush=True)
                name, result = None, parse_error
            else:
                name, result = self._handle_tool_call(tool_call)

            if self.native_tools:
                self.history.append(self.backend.format_tool_call_turn(tool_call or {"name": name, "arguments": {}}))
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

            visible, tool_call, parse_error = self.backend.generate(
                self.history, self.tool_schemas, self.native_tools
            )
            answer += visible

        if tool_call is not None or parse_error is not None:
            msg = "\n[Stopped after too many tool calls in a row without a final answer.]"
            print(msg, end="", flush=True)
            answer += msg

        self.history.append({"role": "assistant", "content": answer})

    def _handle_tool_call(self, tool_call: dict):
        """Execute an already-parsed tool call. Returns (name, result) on
        success, or (name, error_message) on failure — never raises."""
        name = tool_call["name"]
        arguments = tool_call.get("arguments", {})

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

        if name in CONFIRM_REQUIRED and not self._confirm(name, arguments):
            print(f"[skipped: user declined {name}] ", end="", flush=True)
            return name, "The user declined to run this action. Do not attempt it again unless asked."

        print(f"[calling tool: {name}] ", end="", flush=True)
        try:
            result = owner.call_tool(name, arguments)
        except Exception as e:
            return name, f"Tool error: {e}"

        if isinstance(result, dict) and set(result) == {"result"}:
            result = result["result"]
        return name, result

    @staticmethod
    def _confirm(name: str, arguments: dict) -> bool:
        print(f"\n[Confirm] About to call '{name}' with:")
        for key, value in arguments.items():
            if isinstance(value, str) and len(value) > CONFIRM_PREVIEW_CHARS:
                value = value[:CONFIRM_PREVIEW_CHARS] + f"...[{len(value) - CONFIRM_PREVIEW_CHARS} more characters]"
            print(f"    {key}: {value!r}")
        return input("Proceed? [y/N]: ").strip().lower() == "y"
