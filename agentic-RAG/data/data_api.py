
from fastapi import FastAPI, HTTPException, APIRouter
from helpers import load_data, filter_by_date, Datatype
import pandas as pd
import json
"""
Simple API to fetch MES data from external resource
"""



app = FastAPI(title="MES data resource")

print("data", Datatype, Datatype.__mro__)
@app.get("/mes/{data}")
def get_fault_data(data: Datatype, 
                   start_date: str | None = None, 
                   end_date: str | None = None,
                   job_id: str | None = None,
                   part_key: str | None = None,
                   work_center_key: str | None = None,):

    df = load_data(data)
    mask = pd.Series(True, index=df.index)
    if start_date and end_date:
        mask, df = filter_by_date(df, mask, start_date, end_date)
    
    filters = {"job_id": job_id, "part_key": part_key, "work_center_key": work_center_key}
    for name, value in filters.items():
        if value is None:
            continue
        if name not in df.columns:
            return HTTPException(status_code=400, detail=f"Invalid filter: {name}")
        mask &= df[name] == value
    out = df[mask]

    return {"data": data.value,
            "count": len(out),
            "data": json.loads(out.to_json(orient="records", date_format="iso"))} 


