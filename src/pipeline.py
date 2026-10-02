"""รัน pipeline ทั้งหมดด้วยคำสั่งเดียว — ตัวจริงคือ Prefect flow (DAG) ใน src/flow.py

python -m src.pipeline          = ข้อมูลดิบ -> validate -> train -> register(gate) -> export + drift monitors
python -m src.pipeline --serve  = เพิ่ม build/run Docker API ต่อท้าย
"""
from src.flow import main

if __name__ == "__main__":
    main()
