from __future__ import annotations
from typing import Any, Annotated, TypedDict
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from tools import TOOL_SCHEMAS, DISPATCH
from voice_assistant import Speaker, Listener
from config import settings

SYSTEM_PROMPT = """
You are a helpful assistant for shopfloor operators. Answer the questions in the language of the query.
You have access to a vector database of fault codes, instructions, and historical logs. Use this information to answer the questions.

DO NOT make up fault codes or instructions. Only use retrieved information from the vector database or the provided context. If database does not contain relevant information, respond with "I don't know" or "I don't have enough information to answer that."



"""

def _make_llm(with_tools:bool)-> Any:
    """Create a LangChain LLM instance with or without tools."""
    llm = ChatOpenAI(
        base_url=settings.OLLAMA_BASE_URL,
        api_key="ollama",
        model_name=settings.OLLAMA_MODEL,
        temperature=0,
    )

    llm_fallback = ChatOpenAI(
        base_url=settings.OLLAMA_BASE_URL,
        api_key="ollama",
        model_name=settings.OLLAMA_MODEL_FALLBACK,
        temperature=0,
    )

    """if with_tools:
        llm = llm.bind_tools(TOOL_SCHEMAS)
        llm_fallback = llm_fallback.bind_tools(TOOL_SCHEMAS)"""
    return llm.with_fallbacks([llm_fallback])



LLM = _make_llm(with_tools=False)


def chat(query: str, history: list[dict[str, str]] = None) -> dict[str, Any]:
    """Generate a chat response using the LLM."""
    if history is None:
        history = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": query}]
    response = LLM(messages)
    return {"answer": response.content, "messages": messages}

