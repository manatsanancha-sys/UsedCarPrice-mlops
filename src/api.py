from contextlib import asynccontextmanager
from typing import Optional

import cloudpickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.data_cleaning import clean_cars

STATE = {}


class Car(BaseModel):
    name: str
    year: int = Field(ge=1980, le=2026)
    km_driven: int = Field(ge=0, le=3_000_000)
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Optional[str] = None
    engine: Optional[str] = None
    max_power: Optional[str] = None
    torque: Optional[str] = None
    seats: Optional[float] = None


@asynccontextmanager
async def lifespan(app):
    with open("model_export/model.pkl", "rb") as f:
        STATE.update(cloudpickle.load(f))
    yield


app = FastAPI(title="Used Car Price API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": STATE.get("version")}


@app.post("/predict")
def predict(car: Car):
    df = clean_cars(pd.DataFrame([car.model_dump()]))  # ใช้ฟังก์ชัน clean เดียวกับตอนเทรน
    num = STATE["num"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    price = float(STATE["model"].predict(df[STATE["features"]])[0])
    return {"predicted_price_inr": round(price), "model_version": STATE["version"]}
