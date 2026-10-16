
from analytics.load_data import normalize_code
import pandas as pd

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



def routings(df: pd.DataFrame, part_family: str, n: int=3) -> dict[list]:
    df.groupby(["job_id", "op_seq"])

    jobs = df.groupby("job_id").agg(
        route=("work_center_key", tuple),
        part_key=("part_key", "first"),
        batch_qty=("batch_qty","first"),
    )

    total = len(jobs)
    routes = []
    for route, g in jobs.groupby("route"):
        routes.append({"route": list(route),
                       "n_jobs": len(g)})
    routes.sort(key=lambda r: r["n_jobs"], reverse=True)

    return {"part_famliy": part_family, "n_jobs": total,
            "n_distinct_routes": len(routes), "routes": route[:n]}

