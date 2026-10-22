from __future__ import annotations
import json
from typing import Annotated, Any,  TypedDict
import re
from langchain_core.messages import (AIMessage, 
                                     AnyMessage, 
                                     HumanMessage, 
                                     SystemMessage, 
                                    )
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
import operator

from app.tools import tool_retrieve, tool_troubleshoot_lookup, TOOL_SCHEMAS
from app.config import settings



SYSTEM_PROMPT = """
You are a helpful assistant for shopfloor operators. Answer the questions in the language of the query.
DO NOT make up fault codes or instructions. Only use retrieved information and the provided context. 
If database does not contain relevant information, respond with "I don't know" or "I don't have enough information to answer that."

"""

def _make_llm(with_tools:bool)-> Any:
    """Create a LangChain LLM instance with or without tools."""
    llm = ChatOpenAI(
        base_url=settings.OLLAMA_BASE_URL,
        api_key="ollama",
        model_name=settings.OLLAMA_MODEL,
        temperature=0,
    )

    if with_tools:
        llm = llm.bind_tools(TOOL_SCHEMAS)
    return llm


#LLM_TOOLS = _make_llm(with_tools=True)
LLM_PLAIN = _make_llm(with_tools=False)


_MAX_TOOL_RESULT_CHARS = 4000

def _build_question(candidates: list[dict]) -> tuple[str, list[str]]:
    options = [c["symptom"] for c in candidates] + ["None of these"]
    return "Which of these best describes the situation?", options
    
def _resolve_reply(reply: str, options: list[str]) -> int | None:
    m = re.fullmatch(r"\s*(\d+)\s*\.?\s*", reply)
    if m:
        idx = int(m.group(1)) - 1
        return idx if 0 <= idx < len(options) else None
    prompt = (
        "Options:\n" + "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))+
        f"\n\nOperator said: {reply}\nReturn ONLY the number of the matching option, or 0 if none match"
    )
    m = re.search(r"\d+", LLM_PLAIN.invoke(prompt).content or "")
    idx = int(m.group())-1 if m else -1
    return idx if 0 <= idx <len(options) else None


def _format_context():
    ...

def _render_result(result: dict[str, Any]) -> str:
    """Compact JSON rendering of a tool result for the model, size-bounded."""
    try:
        text = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(result)
    if len(text) > _MAX_TOOL_RESULT_CHARS:
        text = text[:_MAX_TOOL_RESULT_CHARS] + " ...[truncated]"
    return text


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    route: str
    query: str
    candidates: list[dict[str, Any]]
    excluded: list[str]
    user_message: str
    context: list[dict[str, Any]]
    answer: str
    


ALARM_RE = re.compile(r"\b(?:\d{4,6}|[AF]\d{2,5}\b)")
TROUBLE_WORDS = ("error", "alarm", "fault", "vika", "hälytys", "virhe")
def _rule_based_route(text: str) -> str:
    if ALARM_RE.search(text) or any(w in text for w in TROUBLE_WORDS):
        return "troubleshoot"
    return "manual"


def classify_node(state: AgentState):
    text = state["messages"][-1].content.lower()
    route = _rule_based_route(text) #NOTE adding possibly text classifier for unclear cases 
    print(f"[ROUTE DEBUG]: Routing -> {route}")
    return {"route": route}

def manuals_node(state: AgentState):
    ret = tool_retrieve(state["user_message"])
    return {"context": ret["result"].get("chunks", [])}
    


def troubleshoot_node(state: AgentState):
    query = state["query"] or state["user_message"]
    candidates = tool_troubleshoot_lookup(query, candidates=state["candidates"])
    return {"query":query, "candidates": candidates, "context": candidates}



def clarify_node(state: AgentState):
    """
    Let LLM ask more clarifying questions from user.
    """
    cands = state["candidates"]
    question, options = _build_question(candidates=cands)
    reply = str(interrupt({"question": question, "options": options}))
    idx = _resolve_reply(reply=reply, options=options)

    update = {"steps": state["steps"]+1,
              "messages": [AIMessage(content=question), HumanMessage(content=reply)] 
              }
    if idx is not None and idx < len(cands):
        update["candidates"] = cands[idx]
        update["resolved"] = True
    else:
        update["query"] = f"{state['query']} {reply}"
        update["excluded"] = state["excluded"]+[c["id"] for c in cands]
    return update


def answer_node(state: AgentState):
    if not state["context"]:
        return {"answer": ""}
    resp = LLM_PLAIN.invoke([SystemMessage(SYSTEM_PROMPT),
                             *state["messages"][-8:],
                             HumanMessage(content=f"Context: \n{_format_context(state['context'])}\n\n"
                                          f"Answer the operator's question using only this context")
                             ])
    answer = (resp.content or "").strip()
    return {"answer": answer, "message": [AIMessage(content=answer)]}


def finalize_node(state: AgentState) -> dict[str, Any]:
    answer = (state.get("answer") or "").strip()
    if not answer:
        answer = (
            "I could not provide a grounded answer from the available tools and data"
            "Please try rephrasing your question"
        )
    return {"answer": answer}

def _route_after_classify(state: AgentState) -> str:
    return state["route"]


def _route_after_troubleshoot(state: AgentState):
    if len(state["candidates"]) > 1 and state["steps"] < settings.MAX_TOOL_ITERS:
        return "clarify"
    return "answer"

def _build_graph()-> Any:
    g = StateGraph(AgentState)
    g.add_node("classify", classify_node)
    g.add_node("manual", manuals_node)
    g.add_node("troubleshoot", troubleshoot_node)
    g.add_node("clarify", clarify_node)
    g.add_node("answer", answer_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"manual": "manual", "troubleshoot": "troubleshoot"}
    )
    g.add_edge("manual", "answer")
    g.add_conditional_edges("troubleshoot",
                            _route_after_troubleshoot,
                            {"clarify": "clarify", "answer": "answer"})
    g.add_edge("clarify", "troubleshoot")
    g.add_edge("answer", "finalize")    
    g.add_edge("finalize", END)
    return g.compile(checkpointer=InMemorySaver())

GRAPH = _build_graph()

def _initial_state(message: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=message)],
        "steps": 0,
        "answer": "",
        "candidates": [],
        "route": "",
        "query": "",
        "context": [],
        "user_message":message,
    }

def generate_chat_response(message: str, session_id: str) -> dict[str, Any]:
    cfg = {"configurable": {"thread_id": session_id}}
    if GRAPH.get_state(cfg).next:
        result = GRAPH.invoke(Command(resume=message), cfg)
    else:
        result = GRAPH.invoke(_initial_state(message=message), cfg)
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return {"answer": payload["question"], "options": payload["options"], "awaiting_reply": True}
    
    return {"answer": result["answer"], "route": result["route"], "sources": result["context"]}
