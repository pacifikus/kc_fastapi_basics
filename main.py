import joblib
import uvicorn

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

with open("elastic_model.pkl", 'rb') as file:
    model = joblib.load(file)


class ModelRequestData(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class Result(BaseModel):
    result: float


@app.get("/")
def root():
    return {"message": "Best regression model"}


@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)


@app.get("/path/{item_id}")
def get_item_path(item_id):
    return {"item_id": item_id}


@app.get("/query/")
async def get_item_query(item_id: int = 0):
    return  {"item_id": item_id}


@app.post("/predict", response_model=Result)
def predict(data: ModelRequestData):
    input_data = data.model_dump()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
