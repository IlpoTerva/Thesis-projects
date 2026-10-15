from pipelines.build_vector_index import build_index
from app.retrieve import embed
from app.chat import chat


def main():
    import sys
    if "--build" in sys.argv:
        
        build_index() #building vector database
    else:
        history = []
        print("Loading vector index and embedding model...")
        embed(["test"])  # warm up the model

        print("Operator assistant ready. Ask a question (Ctrl-C to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q:
                #reply, messages = agent_answer(q, history=history) #ollama chat
                reply = chat(query=q, history=history) #langgraph version
                if "--debug" in sys.argv:
                    print(reply["tool_calls"])
                print("\n--- ANSWER ---")
                ans = reply["answer"]
                print(f"\n{ans}\n")
                history += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()