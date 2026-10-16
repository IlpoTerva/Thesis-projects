from app.tools import (
    tool_get_most_common_codes_API,
    tool_get_date
    )
from analytics.retrieve import retrieve
from typing import Callable, Any

def _r(result: dict[str, any]):
    return {"result": result}

def _missing(param: str):
    return _r({"message": f"Missing required parameter: {param}"})


def get_most_common_codes(type: str, start_date: str, end_date: str, category: str, n: int = 10) -> dict[str, any]:
    """Get the most common fault codes from the MES fault data."""
    if not type:
        return _missing("type")
    if not category:
        return _missing("category")
    result = tool_get_most_common_codes_API(type, start_date, end_date, category, n)
    return _r(result)

def search_documents(query, k=5):
    """Search for documents related to the query using the retrieval system."""
    query = (str(query).strip() if query is not None else "")
    if not query:
        return _missing("query")
    return _r(retrieve(query, k=k))

def get_current_date():
    """Get the current date in YYYY-MM-DD format."""
    return _r({"current_date": tool_get_date()})

DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "search_documents": search_documents,
    "get_most_common_codes_API": get_most_common_codes,
}

def _tool(name: str, description: str, properties: dict[str, any], required: list[str]) -> dict[str, any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }



TOOL_SCHEMAS: list[dict[str, any]] = [
    _tool(
        name="search_documents",
        description="Search for documents related to the query using the retrieval system.",
        properties={
            "query": {"type": "string", "description": "The search query."},
            "k": {"type": "integer", "description": "Number of results to return (default 5)."},
        },
        required=["query"],
    ),
    _tool(
        name="get_most_common_codes_API",
        description="Count the most common values of a field in the MES fault data (e.g. most frequent fault codes, machines, or shifts with faults).",
        properties={
            "type":{
                "type":"string",
                "description": "Type of data fetching from mes can be either 'faults' or 'operations'."
            },
            "start_date": {
                "type": "string",
                "description": "The start date for the data range (YYYY-MM-DD)."
            },
            "end_date": {
                "type": "string",
                "description": "The end date for the data range (YYYY-MM-DD)."
            },
            "category": {
                "type": "string",
                "description": "Which field to count by (a column name, NOT a fault category value). One of: fault_code, fault_category, work_center_key, work_center_type, shift, part_key.",
                },
            "n": {"type": "integer", "description": "Number of top values to return (default 10)."},
            
        },
        required=["type","category"],
    ),

]