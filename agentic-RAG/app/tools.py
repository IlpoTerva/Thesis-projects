from analytics.mes_analytics import (fault_code_lookup, 
                                     most_common_codes, 
                                     fault_code_summary, 
                                     fault_code_summary_API, 
                                     most_common_codes_API,
                                    )
import requests
from app.config import settings
from io import StringIO
import datetime as dt

def tool_get_date():
    """Return the current date in YYYY-MM-DD format."""
    #NOTE : For testing purposes set to a fixed date
    return '2025-10-06' #dt.datetime.now().strftime("%Y-%m-%d") #to get current date

# ---- Tools without API usage ----
def tool_lookup_fault_code(code: str) -> dict:
    """Lookup a fault code in the MES fault codes CSV file. Returns a dictionary of the row if found, else None."""
    return fault_code_lookup(code)

def tool_get_most_common_codes(category: str, n: int = 10, start_date: str = None, end_date: str = None) -> list:
    """Return the n most common codes from the specified category in the MES fault data."""
    return most_common_codes(category, n, start_date, end_date)

def tool_get_fault_code_summary(code: str, start_date: str = None, end_date: str = None) -> dict:
    """Return a summary of a fault code from the MES fault data."""
    return fault_code_summary(code, start_date, end_date)

# ---- API usage tools ----
import pandas as pd
def tool_get_most_common_codes_API(type: str, start_date: str, end_date: str, category: str, n: int = 10) -> list:
    """Return the n most common codes from the specified category in the MES fault data."""
    df = _fetch_mes_data(type, start=start_date, end=end_date)
    #df = pd.read_csv(data, sep=",")
    return most_common_codes_API(df, category, n)

def tool_get_fault_code_summary_API(type: str, start_date: str, end_date: str, code: str) -> dict:
    """Return a summary of a fault code from the MES fault data."""
    df = _fetch_mes_data(type, start=start_date, end=end_date)
    #df = pd.read_csv(data, sep=",")
    return fault_code_summary_API(df, code)


def _fetch_mes_data(type: str, start:str|None=None, end:str|None=None):
    """Fetch MES data from the external API."""
    endpoint = settings.DATA_ENDPOINT + "/mes/"
    resp = requests.get(url=endpoint, params={"type": type, "start_date": start, "end_date": end}).json()
    return pd.read_csv(StringIO(resp["data"]), sep=",")



