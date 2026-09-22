import argparse
import sys
from pathlib import Path

from chat import Chat
from tool_client import ToolClient
from backends.transformers_backend import TransformersBackend
from backends.ollama_backend import OllamaBackend

MCP_SERVER_DIR = Path(__file__).resolve().parent / "mcp_server"

DEFAULT_MODEL_NAME = {
    "transformers": "Qwen/Qwen2.5-3B-Instruct",
    "ollama": "qwen2.5:3b",
}

parser = argparse.ArgumentParser(description="Run the chat application.")
parser.add_argument(
    "--backend", type=str, choices=["transformers", "ollama"], default="ollama",
    help="Which runtime loads and runs the model.",
)
parser.add_argument(
    "--model_name", type=str, default=None,
    help="Model to load. Defaults per backend: a Hugging Face repo id for "
         "transformers (e.g. Qwen/Qwen2.5-3B-Instruct), an Ollama tag for "
         "ollama (e.g. qwen2.5:3b).",
)
parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate.")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")
parser.add_argument("--files_root", type=str, default=".", help="Root directory the filesystem tools may read from.")


def main():
    sys.stdout.reconfigure(errors="replace")
    args = parser.parse_args()
    model_name = args.model_name or DEFAULT_MODEL_NAME[args.backend]

    servers = [
        [str(MCP_SERVER_DIR / "tools_server.py")],
        [str(MCP_SERVER_DIR / "filesystem_server.py"), args.files_root],
        [str(MCP_SERVER_DIR / "apps_server.py")],
    ]

    tool_clients = []
    try:
        for server_args in servers:
            tool_clients.append(ToolClient(command=sys.executable, args=server_args))
    except Exception as e:
        print(f"Failed to start an MCP tool server: {e}")
        for tc in tool_clients:
            tc.close()
        return

    print(f"Using backend={args.backend} model={model_name}")
    if args.backend == "transformers":
        backend = TransformersBackend(model_name, args.max_new_tokens, args.temperature)
    else:
        backend = OllamaBackend(model_name, args.max_new_tokens, args.temperature)

    try:
        chat_app = Chat(backend, tool_clients)
        while True:
            user_input = input("\nUser: ")
            chat_app.chat(user_input)
            if user_input.strip().lower() == "/exit":
                break
    except KeyboardInterrupt:
        print("\nExiting chat...")
    finally:
        for tc in tool_clients:
            tc.close()


if __name__ == "__main__":
    main()
