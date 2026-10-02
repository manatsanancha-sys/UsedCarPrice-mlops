# SLO และนโยบายการเทรนใหม่ (Retrain Policy)

## Service Level Objectives (SLO)

| ตัวชี้วัด | เกณฑ์ | ผลวัดจริง | สถานะ |
|---|---|---|---|
| p50 latency | ≤ 100 ms | 41.9 ms (โหลดพร้อมกัน 10) | ผ่าน |
| p95 latency | ≤ 250 ms | 95.6 ms (โหลดพร้อมกัน 10) | ผ่าน |
| Throughput | ≥ 50 req/s | 211.6 req/s (Docker, 4 workers) | ผ่าน |
| Error rate | ≤ 1% | 0% | ผ่าน |
| Uptime (health check) | ตอบ 200 ภายใน 5s | ผ่าน | ผ่าน |

วัดผ่าน `scripts/load_test.py` และ `src/check_system_health.py` ซึ่งอ่านค่าจริงจาก endpoint `/metrics`

## รูปแบบการให้บริการ (Serving Pattern)

เลือกใช้ **Real-time synchronous serving** ผ่าน FastAPI (ไม่ใช่ batch หรือ async queue) เพราะ:
- ผู้ใช้ (ผู้ขายรถ) ต้องการราคาทันทีตอนกรอกข้อมูล ไม่ใช่รอผลทีหลัง
- โมเดลขนาดเล็ก (Ridge Regression) inference เร็วมาก (< 50ms ต่อคำขอ) ไม่จำเป็นต้องทำ async/batch เพื่อประหยัดทรัพยากร
- ปริมาณคำขอคาดว่าไม่สูงมาก (ผู้ขายรายย่อย ไม่ใช่ระบบ e-commerce ขนาดใหญ่) real-time เดี่ยวรองรับได้เพียงพอ (วัดได้ 211.6 req/s ซึ่งเกินความต้องการจริงมาก)

## นโยบายการเทรนใหม่ (Retrain Policy)

### เกณฑ์ที่กระตุ้นให้เทรนใหม่ (Trigger)
1. **Data Drift:** Evidently ตรวจพบ Dataset Drift = Detected (share of drifted columns > 50%)
2. **Concept Drift:** MAE บนข้อมูลใหม่ (test ปี 2019–2020) หารด้วย `ref_mae` (MAE บน test ปี 2018 ที่บันทึกใน MLflow ตอน register/retrain) เกิน **1.3 เท่า** (`src/monitor_concept_drift.py`) — ใช้ได้ไม่ว่า champion จะเทรนด้วยข้อมูลชุดไหน
3. **กำหนดเวลา:** ตรวจสอบทุก 1 เดือน แม้ไม่มี trigger ข้างต้น (preventive check)

### สถานะปัจจุบันของระบบ (ผลตรวจจริง)
- Data Drift: **Detected** (91.7% ของคอลัมน์)
- Concept Drift: **Detected** (ratio 1.53 เกินเกณฑ์ 1.3)
- **สรุป: เข้าเกณฑ์ต้องเทรนใหม่แล้วตามนโยบาย** (เพราะเราจงใจแบ่งข้อมูลตามปีเพื่อสาธิต drift)

### ขั้นตอนวงจรเทรนใหม่ (Retrain Cycle)
1. **ตรวจพบ** — `src/monitor_drift.py` และ `src/monitor_concept_drift.py` แจ้งเตือน
2. **เทรนใหม่** — รัน `python -m src.retrain` เทรน challenger ด้วยข้อมูลล่าสุด (จำลองด้วย train + val ปี ≤ 2017) แล้ว register เป็นเวอร์ชันใหม่
3. **ประเมิน** — เทียบ MAE/MAPE ของโมเดลใหม่กับโมเดลเดิม (champion ปัจจุบัน) บน test set เดียวกัน
4. **ด่านตรวจ (Gate)** — โมเดลใหม่ต้องมี MAPE ≤ 20% (ตาม Gating metric เดิม) **และ** MAE ไม่แย่กว่า champion
5. **อนุมัติ** — ผ่านทั้งสองข้อ → promote เป็น champion ใหม่ ไม่ผ่าน → คง champion เดิมพร้อมพิมพ์เหตุผล (`src/retrain.py`); จากนั้น `python -m src.export_model` และ build Docker image ใหม่
6. **ย้อนกลับได้เสมอ** — ถ้าโมเดลใหม่แย่กว่าหลัง deploy จริง ใช้ `src/rollback.py` กลับไปเวอร์ชันก่อนหน้าได้ทันที โดยหาเวอร์ชันก่อน champion ที่มี tag `gate_passed=true` อัตโนมัติ ข้ามเวอร์ชันที่ตกด่าน (สาธิตแล้ว: version 3 → version 2 บน registry หลัก และ v3 → v1 ข้าม v2 ที่ตกด่านในการทดสอบ)

## Endpoint ที่เกี่ยวข้อง
- `GET /health` — เช็คว่า service พร้อมใช้งาน
- `GET /metrics` — total_requests, total_errors, p50/p95 latency
- `POST /predict` — ทำนายราคา พร้อม log ทุกคำขอลง `logs/predictions.log`