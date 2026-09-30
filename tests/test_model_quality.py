"""ด่านตรวจคุณภาพโมเดลใน CI: เทรนบนข้อมูลตัวอย่างเล็กๆ แล้วต้องผ่าน gate และชนะ baseline

fixture: tests/fixtures/cars_sample.csv = สุ่ม 1,500 แถวจาก car_details_v3.csv (หลัง dedupe,
random_state=42) แล้วเก็บเฉพาะปี <= 2017 เพื่อแบ่งตามปีแบบเดียวกับ pipeline จริง
"""
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.data_cleaning import clean_cars
from src.data_split import split_by_year
from src.register import GATE_MAPE
from src.train import CAT, MODELS, NUM, TARGET, build
from src.validation import validate

FIXTURE = Path(__file__).parent / "fixtures" / "cars_sample.csv"


@pytest.fixture(scope="module")
def results():
    raw = pd.read_csv(FIXTURE)
    validate(raw)  # fixture ต้องผ่าน schema เดียวกับข้อมูลจริง
    train, val, _ = split_by_year(clean_cars(raw))

    out = {}
    for name, est in [("ridge", Ridge(**MODELS["ridge"].get_params())),
                      ("baseline", DummyRegressor(strategy="median"))]:
        model = build(est).fit(train[NUM + CAT], train[TARGET])
        pred = model.predict(val[NUM + CAT])
        out[name] = {
            "mae": mean_absolute_error(val[TARGET], pred),
            "mape": mean_absolute_percentage_error(val[TARGET], pred),
        }
    return out


def test_ridge_passes_mape_gate(results):
    mape = results["ridge"]["mape"]
    assert mape <= GATE_MAPE, f"MAPE {mape:.1%} เกินเกณฑ์ {GATE_MAPE:.0%}"


def test_ridge_beats_baseline(results):
    assert results["ridge"]["mae"] < results["baseline"]["mae"]
    assert results["ridge"]["mape"] < results["baseline"]["mape"]
