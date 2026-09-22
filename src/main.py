import argparse
import sys
from pathlib import Path

from chat import Chat
from tool_client import ToolClient

MCP_SERVER_DIR = Path(__file__).resolve().parent / "mcp_server"

parser = argparse.ArgumentParser(description="Run the chat application.")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Name of the model to load.")
parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate.")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")
parser.add_argument("--files_root", type=str, default=".", help="Root directory the filesystem tools may read from.")


def main():
    sys.stdout.reconfigure(errors="replace")
    args = parser.parse_args()

    servers = [
        [str(MCP_SERVER_DIR / "tools_server.py")],
        [str(MCP_SERVER_DIR / "filesystem_server.py"), args.files_root],
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

    try:
        chat_app = Chat(args.model_name, tool_clients, args.max_new_tokens, args.temperature)
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
