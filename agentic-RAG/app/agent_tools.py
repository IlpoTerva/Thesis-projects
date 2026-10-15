from app.tools import tool_get_most_common_codes, tool_get_fault_code_summary, tool_lookup_fault_code
from analytics.retrieve import retrieve
from typing import Callable, Any

def _r(result: dict[str, any]):
    return {"result": result}

def _missing(param: str):
    return _r({"message": f"Missing required parameter: {param}"})



def get_most_common_codes(category: str, n: int = 10, start_date: str = None, end_date: str = None) -> dict[str, any]:
    """Get the most common fault codes from the MES fault data."""
    if not category:
        return _missing("category")
    result = tool_get_most_common_codes(category, n, start_date, end_date)
    return _r(result)

def get_fault_code_summary(code: str, start_date: str = None, end_date: str = None) -> dict[str, any]:
    """Get a summary of a specific fault code from the MES fault data."""
    if not code:
        return _missing("code")
    result = tool_get_fault_code_summary(code, start_date, end_date)
    return _r(result)

def lookup_fault_code(code: str) -> dict[str, any]:
    """Look up the title/category/mean resolution time for a fault code from the master reference."""
    if not code:
        return _missing("code")
    result = tool_lookup_fault_code(code)
    return _r(result)

def search_documents(query, k=5):
    """Search for documents related to the query using the retrieval system."""
    query = (str(query).strip() if query is not None else "")
    if not query:
        return _missing("query")
    return _r(retrieve(query, k=k))

DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "search_documents": search_documents,
    "get_most_common_codes": get_most_common_codes,
    "get_fault_code_summary": get_fault_code_summary,
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
        name="get_most_common_codes",
        description="Count the most common values of a field in the MES fault data (e.g. most frequent fault codes, machines, or shifts with faults).",
        properties={
            "category": {
                "type": "string",
                "description": "Which field to count by (a column name, NOT a fault category value). One of: fault_code, fault_category, work_center_key, work_center_type, shift, part_key.",
            },
            "n": {"type": "integer", "description": "Number of top values to return (default 10)."},
            "start_date": {"type": "string", "description": "Start date for filtering (optional, format YYYY-MM-DD)."},
            "end_date": {"type": "string", "description": "End date for filtering (optional, format YYYY-MM-DD)."},
        },
        required=["category"],
    ),
    _tool(
        name="get_fault_code_summary",
        description="Get a statistical summary (count, first/last occurrence, details) of a specific fault code's occurrences in the MES fault data.",
        properties={
            "code": {"type": "string", "description": "The fault code to summarize."},
            "start_date": {"type": "string", "description": "Start date for filtering (optional, format YYYY-MM-DD)."},
            "end_date": {"type": "string", "description": "End date for filtering (optional, format YYYY-MM-DD)."},
        },
        required=["code"],
    ),

]