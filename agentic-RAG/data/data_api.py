from fastapi import FastAPI
from helpers import load_data, filter_by_date
import pandas as pd
"""
Simple API to fetch MES data from external resource
"""

app = FastAPI(title="MES data resource")


@app.get("/mes/")
def get_fault_data(type:str,start_date:str|None = None, end_date:str|None= None):
    path = f"mes/mes_{type}.csv" #faults, operations
    df = load_data(path)
    if start_date and end_date:
        df = filter_by_date(df=df,start=start_date, end=end_date)
    data = pd.DataFrame.to_csv(df)
    return {"data": data} 


