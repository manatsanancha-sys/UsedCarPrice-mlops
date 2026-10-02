"""ทดสอบว่า API รับข้อมูลปกติ และปฏิเสธข้อมูลผิดปกติด้วย 422"""
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import Ridge

from src import api
from src.data_cleaning import clean_cars
from src.train import CAT, NUM, TARGET, build

FIXTURE = Path(__file__).parent / "fixtures" / "cars_sample.csv"

GOOD_CAR = {
    "name": "Maruti Swift Dzire VDI", "year": 2014, "km_driven": 145500,
    "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual",
    "owner": "First Owner", "mileage": "23.4 kmpl", "engine": "1248 CC",
    "max_power": "74 bhp", "torque": "190Nm@ 2000rpm", "seats": 5,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # model_export/ ไม่อยู่ใน repo จึงเทรนโมเดลเล็กจาก fixture แล้วใส่ STATE แทนการโหลดไฟล์
    df = clean_cars(pd.read_csv(FIXTURE))
    model = build(Ridge(alpha=1.0)).fit(df[NUM + CAT], df[TARGET])
    api.STATE.update({"model": model, "features": NUM + CAT, "num": NUM, "version": "test"})
    api.LOG_FILE = tmp_path_factory.mktemp("logs") / "predictions.log"
    return TestClient(api.app)  # ไม่ใช้ with -> ไม่รัน lifespan ที่โหลด model.pkl


def test_valid_car_returns_prices(client):
    r = client.post("/predict", json=GOOD_CAR)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_price_inr"] > 0
    assert body["predicted_price_thb"] > 0


@pytest.mark.parametrize(
    "bad",
    [
        {"fuel": "Water"},
        {"year": 2099},
        {"km_driven": -5},
        {"seats": 100},
        {"seats": 0},
        {"mileage": "abc"},
        {"engine": "-500 CC"},
        {"max_power": "5000 bhp"},
    ],
    ids=["fuel_unknown", "year_out_of_range", "km_negative",
         "seats_too_many", "seats_zero", "mileage_no_number", "engine_negative", "power_too_high"],
)
def test_invalid_car_is_rejected(client, bad):
    r = client.post("/predict", json={**GOOD_CAR, **bad})
    assert r.status_code == 422


def test_numeric_text_error_names_the_field(client):
    r = client.post("/predict", json={**GOOD_CAR, "engine": "-500 CC"})
    assert r.json()["detail"][0]["loc"] == ["body", "engine"]


@pytest.mark.parametrize(
    "ok",
    [
        {"torque": "350Nm@ 1750-2500rpm"},  # ขีดในช่วงรอบไม่ใช่ค่าติดลบ
        {"torque": "22.4 kgm at 1750-2750rpm"},
        {"mileage": None, "engine": None, "max_power": None, "torque": None, "seats": None},
    ],
    ids=["torque_rpm_range", "torque_kgm", "optional_missing"],
)
def test_valid_variants_are_accepted(client, ok):
    assert client.post("/predict", json={**GOOD_CAR, **ok}).status_code == 200


def test_training_rows_pass_api_validation(client):
    # กันการปฏิเสธข้อมูลจริง: ทุกแถวใน fixture (ข้อมูลเทรนจริง) ต้องผ่าน /predict
    rows = pd.read_csv(FIXTURE).drop(columns=[TARGET])
    rows = rows.astype(object).where(rows.notna(), None)
    for row in rows.to_dict("records"):
        r = client.post("/predict", json=row)
        assert r.status_code == 200, (row, r.json())
