from analytics.mes_analytics import (fault_code_lookup, 
                                     most_common_codes, 
                                     fault_code_summary, 
                                     most_common_codes_API,
                                    )
import requests
from app.config import settings
from io import StringIO
import datetime as dt
from typing import Literal, Any


def tool_get_date():
    """Return the current date in YYYY-MM-DD format.
    Usage: Added to system promopt
    """
    #NOTE : For testing purposes set to a fixed date
    return '2025-10-06' #dt.datetime.now().strftime("%Y-%m-%d") #to get current date


# ---- API usage tools ----
import pandas as pd


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
        resp = requests.get(f"{settings.DATA_ENDPOINT}/mes/{type}", params=params)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise MESDataError(f"MES API request failed: {e}") from e
    return pd.DataFrame(resp.json()["data"])

def tool_get_most_common_codes_API(type: str, start_date: str, end_date: str, category: str, n: int = 10) -> dict:
    """Return the n most common codes from the specified category in the MES fault data."""
    try:
        df = _fetch_mes_data(type, start=start_date, end=end_date)
    except MESDataError as e:
        return {"error": str(e)}
    if df.empty:
        return {"n_records": 0, "period": [start_date, end_date], "results": []}
    return {
            "n_records": len(df),
            "period": [start_date, end_date],
            "results": most_common_codes_API(df, category, n)
        }







