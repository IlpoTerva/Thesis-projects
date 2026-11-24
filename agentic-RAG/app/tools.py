from analytics.mes_analytics import (
                                     most_common_codes_API,
                                     routings
                                    )
from analytics.retrieve import retrieve
import requests
from app.config import settings
#import datetime as dt
from typing import Literal, Annotated
import pandas as pd
import inspect
from typing import Callable, Any, get_type_hints
from pydantic import BaseModel, ValidationError, create_model, Field

def get_date():
    """Return the current date in YYYY-MM-DD format.
    Usage: Added to system promopt
    """
    #NOTE : For testing purposes set to a fixed date
    return '2025-10-06' #dt.datetime.now().strftime("%Y-%m-%d") #to get current date

TOOL_SCHEMAS: list[dict[str, Any]] = []
TOOLS: dict[str, tuple[Callable, type[BaseModel]]] = {}

def tool(func: Callable) -> Callable:
    """
    Define tools using decorator
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    fields = {
        name: (hints[name], ... if p.default is inspect.Parameter.empty else p.default)
        for name, p in sig.parameters.items()
    }
    Args = create_model(f"{func.__name__}_args", **fields)
    params = Args.model_json_schema()
    params.pop("title", None)

    TOOL_SCHEMAS.append({"type": "function", "function": {
        "name": func.__name__,
        "description": inspect.getdoc(func),
        "parameters": params
    }})
    TOOLS[func.__name__] = (func, Args)

    return func

def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """
    Call tools
    """
    if name not in TOOLS:
        return {"result": {"error": f"Unknown tool: {name}"}}
    func, Args = TOOLS[name]
    try: 
        validated = Args(**args)
    except ValidationError as e:
        details = "; ".join(
        f"{'.'.join(map(str, err['loc'])) or 'args'}: {err['msg']}" for err in e.errors())
        return {"result": {"error": f"Invalid arguments: {details}. "
                                 "Correct the arguments and call the tool again."}}
    return {"result": func(**validated.model_dump())}


class MESDataError(Exception):
    pass

def _fetch_mes_data(type: Literal["faults", "operations"], **filters) -> pd.DataFrame:
    """Fetch MES data from the external API.
    Params:
        type: 
            fault -> fault MES data
            operations -> MES operations
        filters: filtering parameters
            i.e. start date, end date, job_id, fault_id
    return:
        MES data
    """
    params = {k: v for k, v in filters.items() if v is not None}
    try:
        resp = requests.get(f"{settings.DATA_ENDPOINT}/mes/{type}", params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise MESDataError(f"MES API request failed: {e}") from e
    return pd.DataFrame(resp.json()["data"])

FaultField = Literal["fault_code", "work_center_key", "part_key", "fault_category", "work_center_type"]

@tool
def get_most_common_codes(
    start_date: Annotated[str, Field(description="Start date YYYY-MM-DD, inclusive")], 
    end_date: Annotated[str, Field(description="End date YYYY-MM-DD, inclusive")], 
    group_by: Annotated[list[FaultField]| FaultField, Field(description="Field(s) to group by, e.g. ['fault_code'] or ['work_center_type', 'fault_code']")], 
    metric: Annotated[Literal["count", "duration"], Field(description="'count' how often fault occurs, 'duration' how much downtime fault causes")], 
    n: Annotated[int, Field(ge=1, le=10)] = 10) -> dict:
    """
    Rank fault data by one or more fields, e.g. most frequent fault codes or
    which shifts or work centers lost the most time to faults
    """
    try:
        df = _fetch_mes_data(type="faults", start_date=start_date, end_date=end_date)
    except MESDataError as e:
        return {"error": str(e)}
    if df.empty:
        return {"n_records": 0, "period": [start_date, end_date], "results": []}
    return most_common_codes_API(df, group_by=group_by, metric=metric, n=n)



@tool
def get_routings(
    part_family: Annotated[str, Field(description="Part family, e.g. Turned, SheetMetal, Weldment")], 
    n: Annotated[int, Field(ge=1, le=10)] = 3) -> dict[str, any]:
    """
    Most common historical operation routes for a part family. 
    Use when planning how a new product in that family could be routed.
    """
    try:
        df = _fetch_mes_data(type="operations", part_family=part_family)
    except MESDataError as e:
        return {"error": str(e)}
    if df.empty:
        return{"part_family": part_family, "n_jobs": 0, "routes": []}
    return routings(df, part_family, n)

@tool
def search_documents(query: Annotated[str, Field(min_length=1, description="Search query")],
                          k: Annotated[int, Field(ge=1, le=10)]=5):
    """
    Search fault logs and reference documents from database
    """
    return retrieve(query=query, k=k)




