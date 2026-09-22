class ModelBackend:
    """Interface Chat depends on, so it doesn't need to know which runtime
    (transformers, Ollama, ...) actually produced a reply."""

    def supports_native_tools(self, tool_schemas: list[dict]) -> bool:
        raise NotImplementedError

    def generate(self, history: list[dict], tool_schemas: list[dict], native_tools: bool):
        """Generate one assistant turn, printing visible text live as it's
        produced. Returns (visible_text, tool_call, parse_error):
        - no tool call this turn:            (text, None, None)
        - a tool call was made:              (text, {"name": str, "arguments": dict}, None)
        - a tool call was attempted but
          could not be understood:           (text, None, "<message>")
        """
        raise NotImplementedError

    def format_tool_call_turn(self, tool_call: dict) -> dict:
        """The history entry representing the assistant's own tool-call
        request, in this backend's expected format. Only used when
        native_tools is True."""
        raise NotImplementedError
