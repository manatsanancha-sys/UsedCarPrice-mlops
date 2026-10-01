
"""สาธิตการย้อนกลับ (rollback) โมเดลไปเวอร์ชันก่อนหน้า"""
import sys

import mlflow
from mlflow import MlflowClient

NAME = "used-car-price"

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def main():
    client = MlflowClient()

    versions = sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version))
    print("=== เวอร์ชันทั้งหมดที่มี ===")
    for v in versions:
        print(f"version={v.version}  status={v.status}  gate_passed={v.tags.get('gate_passed', '-')}")

    current = client.get_model_version_by_alias(NAME, "champion")
    print(f"\nตอนนี้ champion คือ version {current.version}")

    # สถานการณ์: champion ปัจจุบันถูก promote แล้วพบว่าแย่กว่าเดิม -> ย้อนไปเวอร์ชันก่อนหน้า
    # ที่เคยผ่านการอนุมัติ (tag gate_passed=true) เท่านั้น ข้ามเวอร์ชันที่ตกด่านแม้จะอยู่ใน registry
    candidates = [v for v in versions
                  if int(v.version) < int(current.version) and v.tags.get("gate_passed") == "true"]
    if not candidates:
        print(f"ERROR: ไม่มีเวอร์ชันก่อน champion (version {current.version}) ที่ผ่าน gate ให้ย้อนกลับ")
        sys.exit(1)
    target_version = str(candidates[-1].version)
    skipped = [str(v.version) for v in versions if int(target_version) < int(v.version) < int(current.version)]
    if skipped:
        print(f"ข้ามเวอร์ชันที่ไม่ผ่าน gate: {', '.join(skipped)}")

    client.set_registered_model_alias(NAME, "champion", target_version)
    print(f"ROLLBACK สำเร็จ -> champion กลับไปที่ version {target_version}")

    confirm = client.get_model_version_by_alias(NAME, "champion")
    print(f"ยืนยัน: champion ตอนนี้คือ version {confirm.version}")


if __name__ == "__main__":
    main()
