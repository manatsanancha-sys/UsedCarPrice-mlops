import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import cloudpickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_cleaning import clean_cars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("used-car-api")

# METRICS อยู่ในหน่วยความจำของ process -> uvicorn --workers N จะมี METRICS แยกกัน N ชุด
# /metrics จึงตอบเฉพาะยอดของ worker ที่รับคำขอนั้น (สุ่มไปมา) ไม่ใช่ยอดรวมทั้งระบบ
# เวลา demo ให้รัน --workers 1 (ดู README หัวข้อ /metrics) ถึงจะเห็นยอดตรงกับจำนวนที่ยิงจริง
METRICS = {"total_requests": 0, "total_errors": 0, "latencies_ms": []}
LOG_FILE = Path("logs/predictions.log")
LOG_FILE.parent.mkdir(exist_ok=True)

# อัตราแลกเปลี่ยน INR -> THB (ปรับตามอัตราปัจจุบันได้)
INR_TO_THB = 0.39

STATE = {}

# ขอบเขตบนของค่าที่แปลงจากข้อความแล้ว (หน่วยหลัง clean_cars: kmpl, CC, bhp, Nm)
# ตั้งให้ครอบคลุมค่าสูงสุดในข้อมูลเทรน (mileage 42, engine 3,604, max_power 400, torque 3,727)
NUMERIC_TEXT_LIMITS = {"mileage": 50, "engine": 10_000, "max_power": 2_000, "torque": 5_000}


class Car(BaseModel):
    name: str
    year: int = Field(ge=1980, le=2026)
    km_driven: int = Field(ge=0, le=3_000_000)
    # ค่าที่ยอมรับต้องตรงกับ schema ใน src/validation.py (ค่าอื่นตอบ 422)
    fuel: Literal["Diesel", "Petrol", "CNG", "LPG"]
    seller_type: Literal["Individual", "Dealer", "Trustmark Dealer"]
    transmission: Literal["Manual", "Automatic"]
    owner: Literal["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
    mileage: Optional[str] = None
    engine: Optional[str] = None
    max_power: Optional[str] = None
    torque: Optional[str] = None
    seats: Optional[float] = Field(default=None, ge=2, le=14)


@asynccontextmanager
async def lifespan(app):
    with open("model_export/model.pkl", "rb") as f:
        STATE.update(cloudpickle.load(f))
    yield


app = FastAPI(title="Used Car Price API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": STATE.get("version")}


def numeric_text_errors(car: Car, cleaned: pd.Series) -> list:
    """ตรวจ mileage/engine/max_power/torque ที่เป็นข้อความ (เช่น "23.4 kmpl") หลังแปลงเป็นตัวเลข
    ไม่ส่งค่า (None) ได้ -> ให้ imputer เติมเหมือนตอนเทรน; แต่ถ้าส่งมาต้องเป็นตัวเลขที่สมเหตุสมผล"""
    errors = []
    for field, limit in NUMERIC_TEXT_LIMITS.items():
        raw = getattr(car, field)
        if raw is None:
            continue
        first_digit = re.search(r"\d", raw)
        if first_digit is None:
            msg = "ไม่พบตัวเลข"
        # clean_cars ไม่อ่านเครื่องหมายลบ ("-500 CC" -> 500) จึงตรวจจากข้อความต้นฉบับ
        # ดูเฉพาะหน้าตัวเลขตัวแรก เพราะ torque ปกติมีช่วงรอบ เช่น "350Nm@ 1750-2500rpm"
        elif raw[: first_digit.start()].rstrip().endswith("-"):
            msg = "ค่าติดลบ"
        elif not 0 <= cleaned[field] <= limit:
            msg = f"เกินขอบเขต 0-{limit:,} (ได้ {cleaned[field]:,.1f})"
        else:
            continue
        errors.append({"loc": ["body", field], "msg": f"{field} ไม่ถูกต้อง: {msg}", "input": raw})
    return errors


@app.post("/predict")
def predict(car: Car):
    df = clean_cars(pd.DataFrame([car.model_dump()]))
    num = STATE["num"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    errors = numeric_text_errors(car, df.iloc[0])
    if errors:
        # ข้อมูลผิดจากฝั่ง client -> 422 แบบเดียวกับ pydantic และไม่นับเป็น error ของระบบใน /metrics
        raise HTTPException(status_code=422, detail=errors)

    start = time.perf_counter()
    METRICS["total_requests"] += 1
    try:
        price = float(STATE["model"].predict(df[STATE["features"]])[0])
        result = {
            "predicted_price_inr": round(price),
            "predicted_price_thb": round(price * INR_TO_THB),
            "model_version": STATE["version"],
        }

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
    # ค่าต่อ worker เท่านั้น (ดูคำอธิบายที่ METRICS); นับเฉพาะคำขอที่เข้าถึง predict() ไม่รวม 422
    lat = sorted(METRICS["latencies_ms"])
    def pct(q):
        return lat[min(len(lat) - 1, int(q * len(lat)))] if lat else None
    return {
        "total_requests": METRICS["total_requests"],
        "total_errors": METRICS["total_errors"],
        "p50_latency_ms": pct(0.5),
        "p95_latency_ms": pct(0.95),
    }

