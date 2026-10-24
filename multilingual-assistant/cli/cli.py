import sys
import uuid

from pipelines.build_vector_index import build_all
from app.retrieve import _embed_query
from app.chat_lang import generate_chat_response

def _new_session() -> str:
    return str(uuid.uuid4())

def main():
    
    if "--build" in sys.argv:
        build_all() #building vector database
        return

    debug = "--debug" in sys.argv
    print("Loading vector index and embedding model...")
    _embed_query("test")  # load up the embedding model
    reset = ["reset", "r"]
    session_id = _new_session()
    print("Operator assistant ready. Ask a question (Ctrl-C to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q in reset:
            session_id = _new_session()
            print("New Session")
        if q not in reset:
            #reply, messages = agent_answer(q, history=history) #ollama chat
            reply = generate_chat_response(message=q, session_id=session_id) #langgraph version
            if debug:
                print(f"[DEBUG] {reply['tool_calls']}") 
            print("\n--- ANSWER ---")
            print(f"[DEBUG ANSWER] {reply}")
            ans = reply["answer"]
            print(f"\n{ans}\n")



if __name__ == "__main__":
    main()