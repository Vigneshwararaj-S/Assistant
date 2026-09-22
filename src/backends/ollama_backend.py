import ollama

from backends.base import ModelBackend


class OllamaBackend(ModelBackend):
    """Talks to a model served by a local Ollama server (a separate
    process, not loaded into this Python process). Only supports models
    with Ollama's native tool-calling support -- there is no raw-text
    fallback path here, unlike TransformersBackend, since tool calls arrive
    as a structured field, never mixed into streamed text."""

    def __init__(self, model_name: str, max_new_tokens: int = 256, temperature: float = 0.7):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.client = ollama.Client()

    def supports_native_tools(self, tool_schemas: list[dict]) -> bool:
        try:
            info = self.client.show(self.model_name)
        except Exception as e:
            raise RuntimeError(
                f"Could not query Ollama for model '{self.model_name}' "
                f"(is it pulled? try: ollama pull {self.model_name}): {e}"
            ) from e
        return "tools" in (info.capabilities or [])

    def generate(self, history: list[dict], tool_schemas: list[dict], native_tools: bool):
        if not native_tools:
            raise RuntimeError(
                f"Model '{self.model_name}' has no native tool-calling support in Ollama; "
                "OllamaBackend only supports tool-capable models."
            )
        stream = self.client.chat(
            model=self.model_name,
            messages=history,
            tools=tool_schemas,
            stream=True,
            options={"temperature": self.temperature, "num_predict": self.max_new_tokens},
        )

        visible_parts = []
        tool_call = None
        for chunk in stream:
            msg = chunk.message
            if msg.tool_calls:
                fn = msg.tool_calls[0].function
                tool_call = {"name": fn.name, "arguments": dict(fn.arguments)}
                continue
            if msg.content:
                print(msg.content, end="", flush=True)
                visible_parts.append(msg.content)

        return "".join(visible_parts), tool_call, None

    def format_tool_call_turn(self, tool_call: dict) -> dict:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": tool_call["name"], "arguments": tool_call["arguments"]}}],
        }
