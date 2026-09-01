import argparse
from chat import Chat

parser = argparse.ArgumentParser(description="Run the chat application.")
parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct", help="Name of the model to load.")
parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate.")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")



def main():
    args = parser.parse_args()
    chat_app = Chat(args.model_name, args.max_new_tokens, args.temperature)
    try:
        while True:
            user_input = input("\nUser: ")
            chat_app.chat(user_input)
            if user_input.strip().lower() == "/exit":
                break
    except KeyboardInterrupt:
        print("\nExiting chat...")


if __name__ == "__main__":
    main()
