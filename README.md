# UsedCarPrice-mlops

ระบบทำนายราคารถมือสอง เพื่อช่วยผู้ขายตั้งราคาที่เหมาะสม
โปรเจกต์รายวิชา CP413008 Machine Learning Engineering for Production

## ภาพรวมระบบ

รับสเปกรถมือสอง (ยี่ห้อ/รุ่น ปี เลขไมล์ เชื้อเพลิง เกียร์ ฯลฯ) แล้วทำนายราคาขาย (รูปีและบาท)
ด้วย Ridge Regression ให้บริการผ่าน REST API (FastAPI) ในคอนเทนเนอร์ Docker
ครอบคลุมวงจร ML ตั้งแต่ข้อมูลดิบ → ตรวจสอบคุณภาพ → เทรน/เทียบโมเดล → ทะเบียนโมเดล + ด่านตรวจ →
ให้บริการ → เฝ้าระวัง Data/Concept Drift → เทรนใหม่/ย้อนกลับ และ CI อัตโนมัติ

| | |
|---|---|
| ข้อมูล | [Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) — 8,128 แถว → 6,926 หลังลบข้อมูลซ้ำ |
| การแบ่งข้อมูล | ตามปีรถ: train ≤ 2016 (5,100) · val = 2017 (808) · test ≥ 2018 (1,018) |
| Optimizing metric | MAE (รูปี) |
| Gating metric | MAPE ≤ 20% บน test set |
| ผลลัพธ์ | pipeline: test MAE 184,906 / MAPE 19.6% · หลัง retrain (train + val): test MAE 177,922 / MAPE 19.2% |

รายละเอียดเชิงธุรกิจอยู่ใน [docs/AI_PROJECT_CANVAS.md](docs/AI_PROJECT_CANVAS.md)
โครงรายงานและ Architecture Diagram อยู่ใน [docs/REPORT_OUTLINE.md](docs/REPORT_OUTLINE.md)

## โครงสร้างโปรเจกต์

```
UsedCarPrice-mlops/
├── .github/workflows/ci.yml     # CI 3 ด้าน: ruff · data/API tests · model quality gate
├── Dockerfile                   # image สำหรับ API (python:3.12-slim, 4 workers, HEALTHCHECK)
├── requirements.txt             # dependency ทั้งหมด ล็อกเวอร์ชัน (ใช้เทรน/ทดสอบ)
├── requirements-api.txt         # dependency เฉพาะ API ใน Docker
├── ruff.toml                    # config ตรวจคุณภาพโค้ด
├── examples/car.json            # ตัวอย่าง request สำหรับ /predict
├── scripts/
│   ├── download_data.py         # ดาวน์โหลดข้อมูลจาก Kaggle -> data/raw/
│   └── load_test.py             # วัด latency p50/p95/p99 + throughput
├── src/
│   ├── data_split.py            # ลบข้อมูลซ้ำ + แบ่ง train/val/test ตามปี
│   ├── validation.py            # Pandera schema
│   ├── data_cleaning.py         # clean_cars() ใช้ร่วมกันทั้งตอนเทรนและให้บริการ
│   ├── train.py                 # เทรน 4 โมเดล + log ลง MLflow
│   ├── register.py              # register โมเดลที่ดีที่สุด + gate MAPE ≤ 20% -> alias champion
│   ├── export_model.py          # export champion -> model_export/model.pkl
│   ├── pipeline.py              # รันทุกขั้นข้างบนต่อกันด้วยคำสั่งเดียว
│   ├── retrain.py               # วงจรเทรนใหม่: challenger vs champion
│   ├── rollback.py              # ย้อน champion ไปเวอร์ชันก่อนหน้าที่ผ่าน gate
│   ├── explain_model.py         # coefficient ของ champion -> docs/model_explanation.md
│   ├── api.py                   # FastAPI: /predict /health /metrics
│   ├── monitor_drift.py         # Data Drift (Evidently) -> reports/drift_report.html
│   ├── monitor_concept_drift.py # Concept Drift (error ratio)
│   └── check_system_health.py   # ตรวจ /health + /metrics เทียบ SLO
├── tests/                       # test_validation · test_api · test_model_quality (+ fixtures/)
├── docs/                        # Canvas · SLO & Retrain Policy · Report Outline · Model Explanation
├── notebooks/                   # พื้นที่สำหรับ notebook สำรวจข้อมูล (ยังว่าง)
└── reports/drift_report.html    # ผล data drift ล่าสุด
```

ไฟล์ที่ระบบสร้างขึ้นเองและไม่อยู่ใน git: `data/`, `mlflow.db`, `mlruns/`, `model_export/`, `logs/`

## วิธีรันจากเครื่องเปล่า

ต้องมี: Python 3.12, Git, Docker (Docker Desktop บน Windows/macOS ต้องเปิดไว้)

### 1. Clone และติดตั้ง

```bash
git clone https://github.com/manatsanancha-sys/UsedCarPrice-mlops.git
cd UsedCarPrice-mlops
python -m venv .venv
```

เปิดใช้ venv — Windows (PowerShell): `.venv\Scripts\Activate.ps1` · Windows (cmd): `.venv\Scripts\activate.bat` ·
macOS/Linux/Git Bash: `source .venv/bin/activate` (Git Bash บน Windows ใช้ `source .venv/Scripts/activate`)

```bash
pip install -r requirements.txt
```

### 2. ดาวน์โหลดข้อมูล

```bash
python scripts/download_data.py
```

ได้ไฟล์ `data/raw/car_details_v3.csv` (ชุดข้อมูลสาธารณะ ไม่ต้องล็อกอิน Kaggle)

### 3. รัน pipeline (split → validation → train → register → export)

```bash
python -m src.pipeline
```

ต้องจบด้วย `=== PIPELINE COMPLETE ===` และได้ `model_export/model.pkl` (champion = version 1)
ถ้าข้อมูลไม่ผ่าน schema หรือโมเดลไม่ผ่าน gate จะหยุดพร้อม `FAILED at: ...` (exit 1)

### 4. เทรนใหม่ (retrain) แล้ว export champion ล่าสุด

```bash
python -m src.retrain
python -m src.export_model
```

`retrain` เทรน challenger ด้วย train + val แล้วเทียบกับ champion บน test set เดียวกัน
promote เมื่อ MAPE ≤ 20% **และ** MAE ไม่แย่กว่า champion (ผลจริง: version 2 ถูก promote, MAE 184,906 → 177,922)
สาธิตกรณีไม่ promote: `python -m src.retrain --alpha 1000`

### 5. Build และรัน API ใน Docker

```bash
docker build -t usedcar-api .
docker run -d --name usedcar-api -p 8000:8000 usedcar-api
```

รอ ~10 วินาทีให้ API พร้อม แล้วทดสอบ (บน PowerShell ใช้ `curl.exe` แทน `curl`):

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @examples/car.json
curl http://localhost:8000/metrics
```

ผลตัวอย่าง `/predict`: `{"predicted_price_inr": ..., "predicted_price_thb": ..., "model_version": "2"}`
เอกสาร API แบบโต้ตอบ: http://localhost:8000/docs

#### ข้อควรรู้เรื่อง `/metrics` (สำคัญตอน demo)

container รัน uvicorn **4 workers** และแต่ละ worker เก็บ `/metrics` ในหน่วยความจำของตัวเอง
ค่าที่ได้จึงเป็นยอดของ worker ที่ตอบคำขอนั้นเท่านั้น — ทดสอบจริง: ยิง `/predict` 20 ครั้ง
แล้วเรียก `/metrics` ซ้ำได้ `total_requests` = 2, 4, 8, 6 สลับกัน (รวมกัน = 20)
และ `check_system_health` อาจผ่านโดยเห็น `total_requests=0` / `p95=None`

**เวลา demo `/metrics` และ `check_system_health` ให้รันด้วย 1 worker** (ยิง 3 ครั้ง → `total_requests=3` ทุกครั้ง):

```bash
docker rm -f usedcar-api
docker run -d --name usedcar-api -p 8000:8000 usedcar-api python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1
```

ตอนวัด throughput (`scripts/load_test.py`) ใช้ 4 workers ตามปกติ — ตัวเลข latency/throughput วัดจากฝั่ง client จึงไม่ได้รับผลกระทบ

ค่าที่ API รับ (ต้องตรงตัวพิมพ์ ค่าอื่นได้ HTTP 422):

| field | ค่าที่รับ |
|---|---|
| `fuel` | `Diesel`, `Petrol`, `CNG`, `LPG` |
| `seller_type` | `Individual`, `Dealer`, `Trustmark Dealer` |
| `transmission` | `Manual`, `Automatic` |
| `owner` | `First Owner`, `Second Owner`, `Third Owner`, `Fourth & Above Owner`, `Test Drive Car` |
| `year` | 1980–2026 |
| `km_driven` | 0–3,000,000 |
| `seats` | 2–14 (ไม่ส่งได้) |
| `mileage`, `engine`, `max_power`, `torque` | ข้อความที่มีตัวเลข เช่น `"23.4 kmpl"`, `"1248 CC"` (ไม่ส่งได้) — ถ้าไม่มีตัวเลข ติดลบ หรือเกิน 50 kmpl / 10,000 CC / 2,000 bhp / 5,000 Nm ได้ 422 |

### 6. วัดประสิทธิภาพและสถานะระบบ (ขณะ container รันอยู่)

```bash
python scripts/load_test.py
python -m src.check_system_health
```

### 7. Monitoring

```bash
python -m src.monitor_drift
python -m src.monitor_concept_drift
```

### 8. สาธิต Rollback

```bash
python -m src.rollback
```

ย้าย alias `champion` ไปเวอร์ชันก่อนหน้าที่**ผ่าน gate** อัตโนมัติ (tag `gate_passed=true`) ข้ามเวอร์ชันที่ตกด่าน เช่น 2 → 1 ถ้าไม่มีเวอร์ชันแบบนั้นจะแจ้ง ERROR (exit 1)
ให้ API ใช้เวอร์ชันที่ย้อนกลับ: `python -m src.export_model` แล้ว build/run Docker ใหม่ (ขั้น 5)

### 9. ทดสอบและตรวจคุณภาพโค้ด (ชุดเดียวกับ CI)

```bash
ruff check .
pytest -v
```

### ปิดระบบ

```bash
docker rm -f usedcar-api
```

## Model Registry / MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

เปิด http://localhost:5000 → experiment `used-car-price` (เปรียบเทียบ run) และ Models → `used-car-price` (เวอร์ชัน + alias)

- ทุก run บันทึก: เวอร์ชันโค้ด (`git_sha`), เวอร์ชันข้อมูล (`data_version` = MD5), hyperparameters, metrics, model artifact, สภาพแวดล้อม (Python/scikit-learn/pandas/numpy)
- `train.py` tune Ridge ด้วย alpha 0.1 / 1 / 10 / 50 / 100 (แยก run `ridge_alpha=…`) เลือกจาก val MAE → alpha = 1.0
- `register.py` เลือก run ที่ val MAE ต่ำสุด (ไม่รวม baseline) → register → ต้องผ่าน gate MAPE ≤ 20% **และ** MAE บน test ไม่แย่กว่า champion ปัจจุบัน จึงตั้ง alias `champion` และติด tag `gate_passed=true`
  - ผ่าน gate แต่แย่กว่า champion → ไม่ promote (`gate_passed=false`) pipeline เดินต่อและ export champion เดิม (exit 0)
  - ไม่ผ่าน gate → `gate_passed=false` และหยุด pipeline (exit 1)
  - ผลคือรัน pipeline ซ้ำได้อย่างปลอดภัย: ถ้าโมเดลใหม่ไม่ดีกว่า champion เดิมจะไม่ถูกแทนที่
- อธิบายผลโมเดล: `python -m src.explain_model` → [docs/model_explanation.md](docs/model_explanation.md) (coefficient รายฟีเจอร์ + เหตุผลการเลือก alpha)
- `export_model.py` export เฉพาะ `champion` ให้ API ใช้

| โมเดล (val set) | MAE | RMSE | MAPE |
|---|---|---|---|
| baseline (median) | 415,937 | 734,079 | 45.3% |
| **Ridge (alpha=1.0)** | **120,615** | **187,566** | 17.1% |
| HistGradientBoosting | 129,150 | 356,127 | 14.4% |
| RandomForest | 135,776 | 363,139 | 15.0% |

## Monitoring

| ด้าน | สคริปต์ | เกณฑ์แจ้งเตือน |
|---|---|---|
| Data Drift | `src/monitor_drift.py` (Evidently) | share of drifted columns > 50% (ผลล่าสุด: 11/12 คอลัมน์ = 91.7%) |
| Concept Drift | `src/monitor_concept_drift.py` | MAE ปัจจุบัน / MAE อ้างอิง > 1.3 |
| System Health | `src/check_system_health.py` | `/health` ไม่ตอบ, p95 latency > 250 ms, error rate > 1% |

นโยบายเทรนใหม่และขั้นตอนเต็ม: [docs/SLO_AND_RETRAIN_POLICY.md](docs/SLO_AND_RETRAIN_POLICY.md)

## SLO

รูปแบบการให้บริการ: Real-time synchronous (ผู้ขายต้องการราคาทันที โมเดลเล็กและเร็ว)
วัดด้วย `scripts/load_test.py` บน Docker 4 workers, พร้อมกัน 10 คำขอ:

| ตัวชี้วัด | เกณฑ์ | วัดได้ |
|---|---|---|
| p50 latency | ≤ 100 ms | 41.9 ms |
| p95 latency | ≤ 250 ms | 95.6 ms |
| Throughput | ≥ 50 req/s | 211.6 req/s |
| Error rate | ≤ 1% | 0% |

## เครื่องมือที่ใช้

| หน้าที่ | เครื่องมือ |
|---|---|
| Version Control | Git + GitHub (feature branch + Pull Request) |
| Containerization | Docker |
| Data Validation | Pandera |
| Experiment Tracking | MLflow Tracking (SQLite backend) |
| Model Registry | MLflow Model Registry (alias `champion`) |
| Pipeline | `src/pipeline.py` |
| Model Serving | FastAPI + uvicorn |
| Monitoring | Evidently (data drift) + สคริปต์ concept drift / system health |
| CI/CD | GitHub Actions |
| Code Quality | ruff |

## การใช้ AI ช่วยพัฒนา

ตามข้อกำหนดรายวิชา ระบุส่วนที่ใช้ AI ช่วย (เครื่องมือ: **Claude Code**, Anthropic)
ทุกส่วนผ่านการรีวิว diff และผลทดสอบโดยทีมก่อน merge และสมาชิกต้องอธิบายโค้ดทุกบรรทัดได้

| Commit | งานที่ AI ช่วย |
|---|---|
| `9e7b706` | ล็อกเวอร์ชัน `requirements.txt` + แก้ `src/pipeline.py` ให้ใช้ `sys.executable` |
| `6117ed9` | CI 3 ด้าน (ruff / data validity / model quality), `ruff.toml`, `tests/test_model_quality.py` + fixture |
| `8c44c83` | ใช้ `Literal` ปฏิเสธค่าหมวดหมู่ผิดใน `src/api.py` + `tests/test_api.py` |
| `c4e6d63` | ร่าง `docs/AI_PROJECT_CANVAS.md` |
| `4c0c249` | ร่าง `docs/REPORT_OUTLINE.md`, หัวข้อนี้ใน README |
| `33d9c02` | `src/retrain.py`, rollback หาเวอร์ชันก่อนหน้าอัตโนมัติ, `register.py` exit 1 เมื่อไม่ผ่าน gate |
| `18e4e67` | เขียน README ใหม่ทั้งไฟล์ + `examples/car.json` (ทดสอบตาม README ใน clone ใหม่) |
| `1357e18` | concept drift ใช้ `ref_mae` ปี 2018 จาก MLflow เทียบกับ MAE ปี 2019–2020 (`register.py`, `retrain.py`, `monitor_concept_drift.py`) |
| `ba82faa` | เกณฑ์แจ้งเตือน data drift 50% + exit code ใน `monitor_drift.py` |
| `7a03d17` | `check_system_health.py` แสดง UNHEALTHY แทน traceback เมื่อ API ไม่ตอบสนอง |
| `593c3fe` | อธิบายข้อจำกัด `/metrics` แบบ multi-worker + คำสั่ง demo `--workers 1` |
| `f98ef3f` | sync ตัวเลขในเอกสารให้ตรงกับ champion v4 และ SLO ล่าสุด |
| `ca9e770` | API ตรวจค่าตัวเลขให้ตรงกับ schema ตอนเทรน (seats 2–14, mileage/engine/max_power/torque) |
| `c0cd1ab` | rollback ข้ามเวอร์ชันที่ตกด่านด้วย tag `gate_passed` |
| `a6b0678` | Ridge hyperparameter tuning (alpha 5 ค่า) + `src/explain_model.py` |
| `b9ea755` | `register.py` เทียบกับ champion ปัจจุบันก่อน promote (ผ่าน gate + MAE ไม่แย่กว่า) |
| — | sync เอกสารกับ tuning, rollback แบบ `gate_passed` และการตรวจค่าตัวเลขของ API |

- AI ช่วยตรวจ repo เทียบกับเกณฑ์การให้คะแนน และคำนวณตัวเลขเชิงธุรกิจจากโมเดลจริง
- ตรวจสอบได้ด้วย `git log --grep="Co-Authored-By: Claude"`

## ทีมพัฒนา

| รหัสนักศึกษา | ชื่อ-สกุล | หน้าที่ |
|---|---|---|
| 673380637-7 | มนัสนันท์ จันดาเวียง | Data pipeline, Model training, MLOps infra, API |
| 673380641-6 | ศาสตรพล อนันเอื้อ | AI Project Canvas (Canva) |
| 673380621-2 | จุฑามาศ ทีหัวช้าง | Architecture Diagram (Canva) |
| 673380333-7 | พิมพ์ศุภา ดำรงคุณาวุฒิ | รายงานหมวด 1–4 |
| 673380649-0 | เพชรการุณย์ มีอุดร | รายงานหมวด 5–8 |
| 673380625-4 | ตันติกร โยทองยศ | ทดสอบ README + รวบรวม Screenshot |
| 673380454-5 | สุธีกานต์ สำราญพัฒน์ | Data Drift alert, Load testing |
