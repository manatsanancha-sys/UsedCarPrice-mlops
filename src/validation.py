import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:
    import pandera as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(nullable=False),
        "year": pa.Column("int64", pa.Check.in_range(1980, 2026)),
        "selling_price": pa.Column("int64", pa.Check.in_range(10_000, 20_000_000)),
        "km_driven": pa.Column("int64", pa.Check.in_range(0, 3_000_000)),
        "fuel": pa.Column(checks=pa.Check.isin(["Diesel", "Petrol", "CNG", "LPG"])),
        "seller_type": pa.Column(
            checks=pa.Check.isin(["Individual", "Dealer", "Trustmark Dealer"])
        ),
        "transmission": pa.Column(checks=pa.Check.isin(["Manual", "Automatic"])),
        "owner": pa.Column(
            checks=pa.Check.isin(
                [
                    "First Owner",
                    "Second Owner",
                    "Third Owner",
                    "Fourth & Above Owner",
                    "Test Drive Car",
                ]
            )
        ),
        "mileage": pa.Column(nullable=True),
        "engine": pa.Column(nullable=True),
        "max_power": pa.Column(nullable=True),
        "torque": pa.Column(nullable=True),
        "seats": pa.Column(float, pa.Check.in_range(2, 14), nullable=True),
    },
    strict=True,  # ห้ามมีคอลัมน์แปลกปลอม
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """ตรวจข้อมูลทั้งก้อน แล้วรายงานทุกข้อผิดพลาดพร้อมกัน (lazy)"""
    return schema.validate(df, lazy=True)


if __name__ == "__main__":
    for name in ["train", "val", "test"]:
        validate(pd.read_csv(f"data/processed/{name}.csv"))
        print(name, "OK")
