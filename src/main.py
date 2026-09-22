import argparse
import sys
from pathlib import Path

from chat import Chat
from tool_client import ToolClient

SERVER_SCRIPT = Path(__file__).resolve().parent / "mcp_server" / "tools_server.py"

parser = argparse.ArgumentParser(description="Run the chat application.")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Name of the model to load.")
parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate.")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")


def main():
    sys.stdout.reconfigure(errors="replace")
    args = parser.parse_args()

    try:
        tool_client = ToolClient(command=sys.executable, args=[str(SERVER_SCRIPT)])
    except Exception as e:
        print(f"Failed to start the MCP tool server: {e}")
        return

    try:
        chat_app = Chat(args.model_name, tool_client, args.max_new_tokens, args.temperature)
        while True:
            user_input = input("\nUser: ")
            chat_app.chat(user_input)
            if user_input.strip().lower() == "/exit":
                break
    except KeyboardInterrupt:
        print("\nExiting chat...")
    finally:
        tool_client.close()


if __name__ == "__main__":
    main()
