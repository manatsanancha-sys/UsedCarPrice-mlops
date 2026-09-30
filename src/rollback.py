
"""สาธิตการย้อนกลับ (rollback) โมเดลไปเวอร์ชันก่อนหน้า"""
from mlflow import MlflowClient

NAME = "used-car-price"


def main():
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{NAME}'")
    print("=== เวอร์ชันทั้งหมดที่มี ===")
    for v in sorted(versions, key=lambda x: int(x.version)):
        print(f"version={v.version}  status={v.status}")

    current = client.get_model_version_by_alias(NAME, "champion")
    print(f"\nตอนนี้ champion คือ version {current.version}")

    # จำลองสถานการณ์: มีเวอร์ชันใหม่ (version 2) ที่แย่กว่า ถูก promote ไปแล้วโดยพลาด
    # แล้วเราตรวจพบว่าแย่กว่า version 1 เดิม เลยต้อง rollback กลับไปที่ version 1

    target_version = "1"
    client.set_registered_model_alias(NAME, "champion", target_version)
    print(f"ROLLBACK สำเร็จ -> champion กลับไปที่ version {target_version}")

    confirm = client.get_model_version_by_alias(NAME, "champion")
    print(f"ยืนยัน: champion ตอนนี้คือ version {confirm.version}")


if __name__ == "__main__":
    main()
