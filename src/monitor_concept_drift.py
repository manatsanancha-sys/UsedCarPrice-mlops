
"""ตรวจ Concept Drift: ความสัมพันธ์ระหว่าง feature กับราคาเปลี่ยนไปหรือไม่
วัดจากการที่ error ของโมเดล (residual) เพิ่มขึ้นเมื่อข้อมูลเปลี่ยนช่วงเวลา
"""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.data_cleaning import clean_cars

NUM = ["year", "km_driven", "mileage", "engine", "max_power", "torque", "seats"]
CAT = ["brand", "fuel", "seller_type", "transmission", "owner"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")

CONCEPT_DRIFT_THRESHOLD = 1.3  # ถ้า error เพิ่มขึ้นเกิน 30% ถือว่า concept drift


def load(name):
    df = clean_cars(pd.read_csv(f"data/processed/{name}.csv"))
    return df[NUM + CAT], df["selling_price"]


def main():
    model = mlflow.sklearn.load_model("models:/used-car-price@champion")

    X_val, y_val = load("val")
    X_test, y_test = load("test")

    val_mae = mean_absolute_error(y_val, model.predict(X_val))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    ratio = test_mae / val_mae

    print(f"val MAE  = {val_mae:,.0f}")
    print(f"test MAE = {test_mae:,.0f}")
    print(f"ratio (test/val) = {ratio:.2f}")

    if ratio > CONCEPT_DRIFT_THRESHOLD:
        print(f"CONCEPT DRIFT DETECTED: error เพิ่มขึ้น {(ratio-1)*100:.0f}% "
              f"(เกินเกณฑ์ {CONCEPT_DRIFT_THRESHOLD}x) -> ความสัมพันธ์ feature-ราคาเปลี่ยนไป "
              f"ควรพิจารณาเทรนโมเดลใหม่")
    else:
        print("ไม่พบ concept drift ที่มีนัยสำคัญ")


if __name__ == "__main__":
    main()
