
"""สาธิตการย้อนกลับ (rollback) โมเดลไปเวอร์ชันก่อนหน้า"""
import sys

import mlflow
from mlflow import MlflowClient

NAME = "used-car-price"

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def main():
    client = MlflowClient()

    versions = {str(v.version): v for v in client.search_model_versions(f"name='{NAME}'")}
    print("=== เวอร์ชันทั้งหมดที่มี ===")
    for v in sorted(versions.values(), key=lambda x: int(x.version)):
        print(f"version={v.version}  status={v.status}")

    current = client.get_model_version_by_alias(NAME, "champion")
    print(f"\nตอนนี้ champion คือ version {current.version}")

    # สถานการณ์: champion ปัจจุบันถูก promote แล้วพบว่าแย่กว่าเดิม -> ย้อนไปเวอร์ชันก่อนหน้า
    target_version = str(int(current.version) - 1)
    if target_version not in versions:
        print(f"ERROR: ไม่มีเวอร์ชันก่อนหน้า champion (version {current.version}) ให้ย้อนกลับ")
        sys.exit(1)

    client.set_registered_model_alias(NAME, "champion", target_version)
    print(f"ROLLBACK สำเร็จ -> champion กลับไปที่ version {target_version}")

    confirm = client.get_model_version_by_alias(NAME, "champion")
    print(f"ยืนยัน: champion ตอนนี้คือ version {confirm.version}")


if __name__ == "__main__":
    main()
