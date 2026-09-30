from contextlib import asynccontextmanager
from typing import Optional

import cloudpickle
import pandas as pd

import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("used-car-api")

METRICS = {"total_requests": 0, "total_errors": 0, "latencies_ms": []}
LOG_FILE = Path("logs/predictions.log")
LOG_FILE.parent.mkdir(exist_ok=True)

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
    start = time.perf_counter()
    METRICS["total_requests"] += 1
    try:
        df = clean_cars(pd.DataFrame([car.model_dump()]))
        num = STATE["num"]
        df[num] = df[num].apply(pd.to_numeric, errors="coerce")
        price = float(STATE["model"].predict(df[STATE["features"]])[0])
        result = {"predicted_price_inr": round(price), "model_version": STATE["version"]}

        latency_ms = (time.perf_counter() - start) * 1000
        METRICS["latencies_ms"].append(latency_ms)

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "input": car.model_dump(),
                "output": result,
                "latency_ms": round(latency_ms, 2),
            }) + "\n")

        return result
    except Exception as e:
        METRICS["total_errors"] += 1
        logger.error(f"predict failed: {e}")
        raise


@app.get("/metrics")
def metrics():
    lat = sorted(METRICS["latencies_ms"])
    def pct(q):
        return lat[min(len(lat) - 1, int(q * len(lat)))] if lat else None
    return {
        "total_requests": METRICS["total_requests"],
        "total_errors": METRICS["total_errors"],
        "p50_latency_ms": pct(0.5),
        "p95_latency_ms": pct(0.95),
    }
