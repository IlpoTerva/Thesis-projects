from pipelines.build_vector_index1 import build_index
from analytics.retrieve import embed
from app.chat_lang import generate_chat_response
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", help="Build the vector index")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")


def main():
    import sys
    args = parser.parse_args()
    if args.build:
        build_index() #building vector database
    else:
        history = []
        reset = ["reset", "clear", "r"]
        print("Loading vector index and embedding model...")
        embed(["test"])  # warm up the model

        print("Operator assistant ready. Ask a question (Ctrl-C to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q in reset:
                history = []
                print("Conversation history cleared.")
            if q not in reset:
                start_time = time.time()
                reply = generate_chat_response(q, history=history) #langgraph version
                if args.debug:
                    print(reply["tool_calls"])
                print("\n--- ANSWER ---")
                ans = reply["answer"]
                print(f"\n{ans}\n")
                print(f"\n[Response time: {time.time() - start_time:.2f} seconds]\n")
                history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]


if __name__ == "__main__":
    main()