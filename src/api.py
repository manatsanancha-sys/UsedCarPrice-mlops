import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Optional

import cloudpickle
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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


# ตัวอย่างรถจริงที่ผ่านการตรวจทุกข้อ -> ใช้เป็นค่าเริ่มต้นใน /docs (กด Execute แล้วได้ 200 ทันที) และในหน้า UI
EXAMPLE_CAR = {
    "name": "Maruti Swift Dzire VDI", "year": 2014, "km_driven": 145500,
    "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner",
    "mileage": "23.4 kmpl", "engine": "1248 CC", "max_power": "74 bhp", "torque": "190Nm@ 2000rpm", "seats": 5,
}


class Car(BaseModel):
    model_config = {"json_schema_extra": {"examples": [EXAMPLE_CAR]}}

    name: str = Field(description="ชื่อยี่ห้อและรุ่น (คำแรกใช้เป็นยี่ห้อ)", examples=["Maruti Swift Dzire VDI"])
    year: int = Field(ge=1980, le=2026, description="ปีรถ (1980–2026)", examples=[2014])
    km_driven: int = Field(ge=0, le=3_000_000, description="เลขไมล์ หน่วยกิโลเมตร (0–3,000,000)",
                           examples=[145500])
    # ค่าที่ยอมรับต้องตรงกับ schema ใน src/validation.py (ค่าอื่นตอบ 422)
    fuel: Literal["Diesel", "Petrol", "CNG", "LPG"] = Field(description="เชื้อเพลิง", examples=["Diesel"])
    seller_type: Literal["Individual", "Dealer", "Trustmark Dealer"] = Field(
        description="ประเภทผู้ขาย: Individual = บุคคล, Dealer / Trustmark Dealer = ดีลเลอร์", examples=["Individual"])
    transmission: Literal["Manual", "Automatic"] = Field(description="เกียร์: Manual = ธรรมดา, Automatic = อัตโนมัติ",
                                                         examples=["Manual"])
    owner: Literal["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"] = Field(
        description="ลำดับเจ้าของ เช่น First Owner = มือหนึ่ง", examples=["First Owner"])
    mileage: Optional[str] = Field(default=None, description="อัตราสิ้นเปลือง หน่วย kmpl (ไม่ส่งได้)",
                                   examples=["23.4 kmpl"])
    engine: Optional[str] = Field(default=None, description="ขนาดเครื่องยนต์ หน่วย CC (ไม่ส่งได้)",
                                  examples=["1248 CC"])
    max_power: Optional[str] = Field(default=None, description="กำลังสูงสุด หน่วย bhp (ไม่ส่งได้)",
                                     examples=["74 bhp"])
    torque: Optional[str] = Field(default=None, description="แรงบิด หน่วย Nm หรือ kgm (ไม่ส่งได้)",
                                  examples=["190Nm@ 2000rpm"])
    seats: Optional[float] = Field(default=None, ge=2, le=14, description="จำนวนที่นั่ง 2–14 (ไม่ส่งได้)",
                                   examples=[5])


class Prediction(BaseModel):
    # ลำดับฟิลด์ตรงกับ response เดิม (inr, thb, model_version)
    predicted_price_inr: int = Field(description="ราคาที่ทำนาย หน่วยรูปีอินเดีย (ผลจากโมเดลโดยตรง)",
                                     examples=[437799])
    predicted_price_thb: int = Field(description="ราคาที่ทำนาย หน่วยบาท (แปลงจากรูปีที่ 1 INR = 0.39 THB)",
                                     examples=[170741])
    model_version: str = Field(description="เวอร์ชันโมเดล (champion) ที่ใช้ทำนาย", examples=["4"])


@asynccontextmanager
async def lifespan(app):
    with open("model_export/model.pkl", "rb") as f:
        STATE.update(cloudpickle.load(f))
    yield


# /docs แบบออฟไลน์: ปิด docs เริ่มต้น (โหลด Swagger UI จาก CDN) แล้วเสิร์ฟไฟล์จาก static/ ของโปรเจกต์แทน
# ตามแนวทาง "Self-hosting JavaScript and CSS for docs" ของ FastAPI — ไฟล์และเวอร์ชันดูใน static/README.md
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

OPENAPI_TAGS = [
    {"name": "ทำนายราคา", "description": "ส่งสเปกรถมือสอง แล้วได้ราคาที่เหมาะสม (บาทและรูปี)"},
    {"name": "เฝ้าระวังระบบ", "description": "ตรวจว่า API พร้อมใช้งานและดูสถิติการใช้งาน"},
]

app = FastAPI(title="Used Car Price API", lifespan=lifespan, docs_url=None, openapi_tags=OPENAPI_TAGS,
              description="ระบบทำนายราคารถมือสอง — หน้าใช้งานภาษาไทยอยู่ที่ `/` · เอกสารนี้ใช้งานออฟไลน์ได้")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/docs", include_in_schema=False)
def swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.png",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/", include_in_schema=False)
@app.get("/app", include_in_schema=False)
def thai_ui():
    # หน้า UI ภาษาไทยสำหรับสาธิต (ไฟล์เดียว ไม่ใช้ CDN) เรียก /predict ด้วย relative path
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health", tags=["เฝ้าระวังระบบ"], summary="ตรวจสถานะระบบ (Health Check)",
         description="บอกว่า API พร้อมใช้งานไหม และโมเดลเวอร์ชันอะไร")
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


@app.post("/predict", tags=["ทำนายราคา"], summary="ทำนายราคารถมือสอง", response_model=Prediction,
          description="ส่งสเปกรถ แล้วได้ราคาที่ทำนายเป็นบาทและรูปี ข้อมูลผิดรูปแบบหรือเกินขอบเขตตอบ 422 "
                      "พร้อมบอกช่องที่ผิด",
          responses={422: {"description": "ข้อมูลไม่ถูกต้อง — `detail` บอกช่อง (`loc`) และเหตุผล (`msg`)"}})
def predict(car: Annotated[Car, Body(openapi_examples={
        "maruti": {"summary": "Maruti Swift Dzire VDI ปี 2014 (ข้อมูลจริง)", "value": EXAMPLE_CAR}})]):
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


@app.get("/metrics", tags=["เฝ้าระวังระบบ"], summary="ดูสถิติการใช้งาน (Metrics)",
         description="จำนวนคำขอ ข้อผิดพลาด และความเร็วตอบกลับ (หมายเหตุ: นับแยกตาม worker)")
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

