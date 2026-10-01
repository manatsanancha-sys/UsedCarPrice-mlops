"""ตรวจ Data Drift ด้วย Evidently: เทียบการกระจายของ feature ระหว่างข้อมูลอ้างอิงกับข้อมูลปัจจุบัน

exit 0 = ไม่พบ drift, exit 1 = share of drifted columns เกินเกณฑ์ (ตาม docs/SLO_AND_RETRAIN_POLICY.md)
"""
import argparse
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.data_cleaning import clean_cars

NUM = ["year", "km_driven", "mileage", "engine", "max_power", "torque", "seats"]
CAT = ["brand", "fuel", "seller_type", "transmission", "owner"]

DATA_DRIFT_THRESHOLD = 0.5  # ถ้าคอลัมน์ที่ drift เกิน 50% ถือว่า dataset drift


def drifted_columns(result):
    """อ่าน (จำนวน, สัดส่วน) คอลัมน์ที่ drift จากผล DataDriftPreset"""
    for m in result.dict()["metrics"]:
        if m["metric_name"].startswith("DriftedColumnsCount"):
            return int(m["value"]["count"]), m["value"]["share"]
    raise RuntimeError("ไม่พบ DriftedColumnsCount ในผล Evidently")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/processed/train.csv")
    parser.add_argument("--current", default="data/processed/test.csv")
    args = parser.parse_args()

    ref = clean_cars(pd.read_csv(args.reference))[NUM + CAT]
    cur = clean_cars(pd.read_csv(args.current))[NUM + CAT]

    report = Report([DataDriftPreset(drift_share=DATA_DRIFT_THRESHOLD)])
    result = report.run(reference_data=ref, current_data=cur)
    result.save_html("reports/drift_report.html")
    print("saved reports/drift_report.html")

    count, share = drifted_columns(result)
    total = len(NUM + CAT)
    print(f"reference = {args.reference} ({len(ref)} แถว), current = {args.current} ({len(cur)} แถว)")
    print(f"drifted columns = {count}/{total} (share {share:.1%}, เกณฑ์ {DATA_DRIFT_THRESHOLD:.0%})")

    if share > DATA_DRIFT_THRESHOLD:
        print(f"DATA DRIFT DETECTED: {count}/{total} คอลัมน์ ({share:.1%}) drift เกินเกณฑ์ "
              f"{DATA_DRIFT_THRESHOLD:.0%} -> การกระจายของข้อมูลเปลี่ยนไป ควรพิจารณาเทรนโมเดลใหม่")
        sys.exit(1)
    print("ไม่พบ data drift ที่เกินเกณฑ์")


if __name__ == "__main__":
    main()
