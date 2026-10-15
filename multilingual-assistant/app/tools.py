from retrieve import retrieve
from typing import Callable, Any

def _r(result: dict[str, any]):
    return {"result": result}

def _missing(param: str):
    return _r({"message": f"Missing required parameter: {param}"})

def tool_retrieve(query, k=5):
    """
    Tool function to retrieve relevant information based on a query.
    This function can be used as a tool in the chat pipeline.
    """
    query = (str(query).strip() if query is not None else "")
    if not query:
        return _missing("query")
    return _r(retrieve(query, k=k))

DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "retrieve": tool_retrieve,
}


def _tool(name: str, description: str, properties: dict[str, any], required: list[str]) -> dict[str, any]:
    """
    Helper function to create a tool schema.
    """
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


TOOL_SCHEMAS: list[dict[str, any]] = [
    _tool(
        name="retrieve",
        description="Retrieve relevant information based on a query.",
        properties={
            "query": {"type": "string", "description": "The query to retrieve information for."},
            "k": {"type": "integer", "description": "The number of results to retrieve (default is 5)."},
        },
        required=["query"],
    ),
]