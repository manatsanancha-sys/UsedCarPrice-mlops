# UsedCarPrice-mlops

ระบบทำนายราคารถมือสอง เพื่อช่วยผู้ขายตั้งราคาที่เหมาะสม
โปรเจกต์รายวิชา CP413008 Machine Learning Engineering for Production

## ภาพรวมระบบ

ระบบนี้รับข้อมูลสเปกรถมือสอง (ยี่ห้อ/รุ่น ปี เลขไมล์ เชื้อเพลิง เกียร์ ฯลฯ) แล้วทำนายราคาขายที่เหมาะสม
โดยใช้โมเดล Ridge Regression ให้บริการผ่าน REST API (FastAPI) ที่รันในคอนเทนเนอร์ Docker
พร้อมระบบเฝ้าระวัง Data Drift / Concept Drift และ CI/CD อัตโนมัติ ครอบคลุมวงจรชีวิตของ ML ตั้งแต่
รับข้อมูลดิบ ทำความสะอาด ตรวจสอบคุณภาพ เทรน เปรียบเทียบโมเดล ขึ้นทะเบียน ให้บริการ จนถึงเฝ้าระวัง

**Optimizing metric:** MAE (Mean Absolute Error)
**Gating metric:** MAPE ≤ 20% บน test set

**ผลล่าสุด:** Test MAE = 184,906 รูปี, Test MAPE = 19.6% (ผ่านเกณฑ์ gate)

## โครงสร้างโปรเจกต์

UsedCarPrice-mlops/
├── src/
│ ├── data_cleaning.py # ทำความสะอาดข้อมูล (ใช้ร่วมกันทั้ง train และ serve)
│ ├── data_split.py # แบ่งข้อมูล train/val/test ตามปี (จำลอง drift)
│ ├── validation.py # ตรวจสอบ schema ด้วย Pandera
│ ├── train.py # เทรนโมเดล 4 แบบ บันทึกผลลง MLflow
│ ├── register.py # ขึ้นทะเบียนโมเดลที่ดีที่สุด + ด่านตรวจก่อน promote
│ ├── rollback.py # สาธิตการย้อนกลับเวอร์ชันโมเดล
│ ├── export_model.py # เตรียมโมเดล champion สำหรับ API
│ ├── api.py # FastAPI serving (/health, /predict, /metrics)
│ ├── monitor_drift.py # ตรวจ Data Drift ด้วย Evidently
│ ├── monitor_concept_drift.py # ตรวจ Concept Drift จาก error ratio
│ ├── check_system_health.py # ตรวจสถานะระบบเทียบกับ SLO
│ └── pipeline.py # รวมทุกขั้นตอนเป็น pipeline เดียว
├── tests/ # เทสต์อัตโนมัติ (pandera schema, ข้อมูลเสีย)
├── scripts/
│ ├── download_data.py # โหลดข้อมูลจาก Kaggle
│ └── load_test.py # ทดสอบ latency/throughput ของ API
├── docs/
│ └── SLO_AND_RETRAIN_POLICY.md
├── reports/
│ └── drift_report.html # รายงาน Data Drift (Evidently)
├── .github/workflows/ci.yml # GitHub Actions CI
├── Dockerfile
├── requirements.txt
├── requirements-api.txt
└── pytest.ini
