"""
File to test tool calls and retrieval functions.
"""
from app.tools import tool_get_mes_data


resp= tool_get_mes_data("faults", "2025-02-10", "2025-02-16")
print(resp)