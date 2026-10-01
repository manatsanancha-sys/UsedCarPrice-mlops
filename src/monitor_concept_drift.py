
"""ตรวจ Concept Drift: ความสัมพันธ์ระหว่าง feature กับราคาเปลี่ยนไปหรือไม่
เทียบ MAE บนข้อมูลใหม่ (test ปีหลัง REF_YEAR) กับ ref_mae ที่บันทึกไว้ใน MLflow run ของ champion
(MAE บน test ปี REF_YEAR ตอน register) -> ใช้ได้ไม่ว่า champion จะเทรนด้วยข้อมูลชุดไหน

exit 0 = ไม่พบ drift, exit 1 = พบ drift, exit 2 = champion ไม่มี ref_mae (ประเมินไม่ได้)
"""
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from src.data_cleaning import clean_cars

NUM = ["year", "km_driven", "mileage", "engine", "max_power", "torque", "seats"]
CAT = ["brand", "fuel", "seller_type", "transmission", "owner"]
NAME = "used-car-price"
REF_YEAR = 2018  # ต้องตรงกับ REF_YEAR ใน src/register.py

mlflow.set_tracking_uri("sqlite:///mlflow.db")

CONCEPT_DRIFT_THRESHOLD = 1.3  # ถ้า error เพิ่มขึ้นเกิน 30% ถือว่า concept drift


def main():
    client = MlflowClient()
    champ = client.get_model_version_by_alias(NAME, "champion")
    params = client.get_run(champ.run_id).data.params
    print(f"champion = version {champ.version}")

    if "ref_mae" not in params:
        print("champion เวอร์ชันนี้ไม่มี ref_mae บันทึกไว้ ต้อง retrain ใหม่ด้วยโค้ดเวอร์ชันล่าสุดก่อน")
        sys.exit(2)
    ref_mae = float(params["ref_mae"])

    model = mlflow.sklearn.load_model(f"models:/{NAME}/{champ.version}")
    df = clean_cars(pd.read_csv("data/processed/test.csv"))
    cur = df[df["year"] > REF_YEAR]
    cur_mae = mean_absolute_error(cur["selling_price"], model.predict(cur[NUM + CAT]))
    ratio = cur_mae / ref_mae

    years = f"{cur['year'].min()}-{cur['year'].max()}"
    print(f"ref MAE ({REF_YEAR}, จาก MLflow) = {ref_mae:,.0f}")
    print(f"current MAE ({years}, {len(cur)} แถว) = {cur_mae:,.0f}")
    print(f"ratio (current/ref) = {ratio:.2f}")

    if ratio > CONCEPT_DRIFT_THRESHOLD:
        print(f"CONCEPT DRIFT DETECTED: error เพิ่มขึ้น {(ratio-1)*100:.0f}% "
              f"(เกินเกณฑ์ {CONCEPT_DRIFT_THRESHOLD}x) -> ความสัมพันธ์ feature-ราคาเปลี่ยนไป "
              f"ควรพิจารณาเทรนโมเดลใหม่")
        sys.exit(1)
    print("ไม่พบ concept drift ที่มีนัยสำคัญ")


if __name__ == "__main__":
    main()
