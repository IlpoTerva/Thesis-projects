
from analytics.load_data import load_fault_codes, load_mes_fault_data, normalize_code
import pandas as pd


# --- Tools without external API calling ----

def filter_data_by_date(df, start_date: str = None, end_date: str = None):
    """Filter a DataFrame by start and end dates."""
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]
    return df

def fault_code_lookup(code: str) -> dict:
    """Lookup a fault code in the MES fault codes CSV file. Returns a dictionary of the row if found, else None."""
    lookup, _ = load_fault_codes()
    return lookup.get(normalize_code(code))

def most_common_codes(category: str, n: int = 10, start_date: str = None, end_date: str = None):
    """Return the n most common values for a given field (column) in the MES fault data,
    i.e most common machines that have fault codes, operations that have fault codes, etc.
    Returns a dict of {value: count}, or {"error": ...} if `category` is not a valid field.
    """
    df = load_mes_fault_data()
    if df.empty:
        return {}
    if category not in df.columns:
        return {"error": f"'{category}' is not a valid field. Valid fields: {list(df.columns)}"}
    if start_date and end_date:
        df = filter_data_by_date(df, start_date, end_date)
    return df[category].value_counts().head(n).to_dict()


def fault_code_summary(code: str, start_date: str = None, end_date: str = None) -> dict:
    """Return a summary of a fault code from the MES fault data."""
    df = load_mes_fault_data()
    if df.empty or "fault_code" not in df.columns:
        return {}
    df = filter_data_by_date(df, start_date, end_date)
    code_norm = normalize_code(code)
    code_df = df[df["fault_code"].apply(normalize_code) == code_norm]
    if code_df.empty:
        return {}
    return {
        "code": code,
        "count": len(code_df),
        "first_occurrence": code_df["date"].min(),
        "last_occurrence": code_df["date"].max(),
        "details": code_df.to_dict(orient="records"),
    }



# ---- Tools if API is utilized to get MES data ----

def most_common_codes_API(df: pd.DataFrame, category: str, n: int = 10):
    """Return the n most common values for a given field (column) in the MES fault data,
    i.e most common machines that have fault codes, operations that have fault codes, etc.
    Returns a dict of {value: count}, or {"error": ...} if `category` is not a valid field.
    """
    
    return df[category].value_counts().head(n).to_dict()


def fault_code_summary_API(df:pd.DataFrame,code: str) -> dict:
    """Return a summary of a fault code from the MES fault data."""    
    code_norm = normalize_code(code)
    code_df = df[df["fault_code"].apply(normalize_code) == code_norm]
    if code_df.empty:
        return {}
    return {
        "code": code,
        "count": len(code_df),
        "first_occurrence": code_df["date"].min(),
        "last_occurrence": code_df["date"].max(),
        "details": code_df.to_dict(orient="records"),
    }

