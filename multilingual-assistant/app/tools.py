from app.retrieve import retrieve
from typing import Callable, Any
from langchain.tools import tool


def _r(result: dict[str, any]):
    return {"result": result}

def _missing(param: str):
    return _r({"message": f"Missing required parameter: {param}"})


@tool
def tool_retrieve(query, k=5):
    """
    Tool function to retrieve relevant information based on a query.
    This function can be used as a tool in the chat pipeline.
    """
    query = (str(query).strip() if query is not None else "")
    if not query:
        return _missing("query")
    return _r(retrieve(query, k=k))

@tool
def diagnostic_troubleshoot_tool(symptom: str):
    """
    Use this tool ONLY when a machine fails, throws an error/alarm code,
    makes strange noises, or breaks down mechanically.
    """
    ...

DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "retrieve": tool_retrieve,
}


