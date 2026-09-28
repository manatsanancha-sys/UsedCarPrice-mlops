import pandas as pd
import pytest

try:
    from pandera.errors import SchemaErrors
except ImportError:
    from pandera import SchemaErrors

from src.validation import validate


def good_row():
    return {
        "name": "Maruti Swift Dzire VDI",
        "year": 2014,
        "selling_price": 450000,
        "km_driven": 145500,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage": "23.4 kmpl",
        "engine": "1248 CC",
        "max_power": "74 bhp",
        "torque": "190Nm@ 2000rpm",
        "seats": 5.0,
    }


def make_df(**overrides):
    row = good_row()
    row.update(overrides)
    return pd.DataFrame([row])


def test_good_data_passes():
    validate(make_df())


@pytest.mark.parametrize(
    "bad",
    [
        {"selling_price": -1},
        {"year": 2099},
        {"km_driven": -5},
        {"fuel": "Water"},
    ],
)
def test_bad_data_is_caught(bad):
    with pytest.raises(SchemaErrors):
        validate(make_df(**bad))
