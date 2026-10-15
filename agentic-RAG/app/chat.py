
from app.config import settings
from app.agent_tools import TOOL_SCHEMAS
from app.agent_tools import search_documents, get_most_common_codes, get_fault_code_summary, lookup_fault_code
import json


SYSTEM_PROMPT = """ 
You are an assistant for factory-floor operators. Answer concisely and practically, in the operator's language (Finnish or English, matching the question). 
You answer ONLY from tools. You have two kind of tools:
- Analytics tools: you can search for fault codes and related instructions from the MES fault data
- Retrieval tools: you can search for past incidents and similar cases from the factory's knowledge base

ONLY USE the tools to answer the question. DO NOT make up answers or use ANY other knowledge.
DO NOT guess if the answer is not found in the tools. If you cannot find an answer, say plainly that no relevant records were found.



Rules:
- For mixed questions use both kind of tools (e.g. "Is there some fault code in mes and has it happened before?"). 
- WHEN using both kind of tools, First check the MES and THEN search documents for similar cases.
- If a fault code is mentioned and mes is not mentioned explicitly in the question, retrieve only.
- If a fault code is mentioned and mes is explicitly mentioned in the question, use analytics tools.

If a past incident or reference shows a fix that worked, lead with it ONLY if the user asks for it. 
If the context contains no recommended action or fix, say plainly that no recorded fix was found — do NOT invent steps.

DO NOT add anything to the answer that is not in the tools. Do NOT make up any steps or instructions.

Remember: Use ONLY the provided tools for the answer. 

"""
TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "get_most_common_codes": get_most_common_codes,
    "get_fault_code_summary": get_fault_code_summary,
}

def dispatch(name, args):
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**(args or {}))
    except Exception as e:
        return {"error": f"{name} failed: {e}"}


def ollama_chat(messages, models=None, tools=None):
    import requests
    models = models or settings.LLM_MODELS
    last_err = None
    for model in models:
        try:
            r = requests.post(
                settings.OLLAMA_BASE_URL,          # http://localhost:11434/api/chat
                json={"model": model, "messages": messages, "stream": False,
                      "tools": tools, "options": {"temperature": 0.2}},
                
            )
            r.raise_for_status()
            data = r.json()
            print(f"[debug] response: {data}")
            if data.get("error"):
                raise RuntimeError(data["error"])
            return data["message"]                 # {role, content, tool_calls?}
        except Exception as e:
            last_err = e
            print(f"[llm] {model} failed ({e}); falling back...")
    raise RuntimeError(f"All models failed. Last error: {last_err}")


def agent_answer(query, history=None, max_steps=4):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += (history or [])
    messages.append({"role": "user", "content": query})
    for _ in range(max_steps):
        msg = ollama_chat(messages, models=settings.LLM_MODELS, tools=TOOL_SCHEMAS)
        messages.append(msg)

        calls = msg.get("tool_calls")
        print(f"[debug] tool calls: {calls}")
        if not calls:
            return msg.get("content", ""), messages   # model gave a final answer

        for call in calls:                              # execute each requested tool
            fn = call["function"]
            result = dispatch(fn["name"], fn.get("arguments"))
            print(f"[debug] tool result: {result}")
            messages.append({"role": "tool", "content": json.dumps(result, default=str)})

    return "Reached step limit without a final answer.", messages