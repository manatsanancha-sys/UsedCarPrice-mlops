# โครงรายงาน + คำอธิบาย Architecture Diagram (ร่าง)

> โครงสำหรับเขียนรายงานฉบับเต็มและวาดแผนภาพใน Canva เรียงตามเกณฑ์การให้คะแนน 8 หมวด
> ตัวเลขทุกตัวตรวจจากไฟล์/ผลรันจริงใน repo ณ commit `8e2755c` (main) เว้นแต่ติดป้ายไว้ดังนี้
> - **[จากเอกสารเดิม]** — ยกมาจาก `docs/SLO_AND_RETRAIN_POLICY.md` ไม่ได้วัดซ้ำตอนร่างเอกสารนี้
> - **[TODO]** — ยังไม่มีในโปรเจกต์ ต้องทำหรือเขียนเพิ่มก่อนส่ง
>
> ราคาเป็นรูปี (INR) · 1 INR = 0.39 THB (`INR_TO_THB` ใน `src/api.py`)

---

## 0. บทนำ / ภาพรวม
- ปัญหา: ผู้ขายรถมือสองตั้งราคาไม่ถูก → ขาดทุนหรือขายช้า (รายละเอียดใน `docs/AI_PROJECT_CANVAS.md`)
- ผลลัพธ์หลัก: Ridge Regression, test MAE 184,906 INR (~72,100 THB), test MAPE 19.6% ผ่าน gate ≤ 20%
- ระบบครบวงจร: ข้อมูลดิบ → ตรวจสอบ → เทรน → ทะเบียนโมเดล → API ใน Docker → เฝ้าระวัง → CI
- สมาชิกกลุ่มและหน้าที่ **[TODO — ใส่ชื่อ/รหัส/งานที่รับผิดชอบ]**

---

## หมวด 1: การวางกรอบปัญหาและการวางแผน (2 คะแนน)
- **AI Project Canvas** — เนื้อหาครบใน `docs/AI_PROJECT_CANVAS.md` (Value Proposition, Customers, Stakeholders, ML Task, Data, Output/Integration, Metrics, Cost, Revenue) → นำภาพจาก Canva มาใส่
- **ทำไมใช้ ML แทนกฎ** — baseline "ราคากลางของตลาด" (`DummyRegressor(strategy="median")`) ได้ val MAPE 45.3% vs Ridge 17.1%; ข้อมูลมี 32 ยี่ห้อ × เชื้อเพลิง × เกียร์ × เจ้าของ เขียนกฎเองไม่ไหว
- **Optimizing metric = MAE** — อ่านเป็นเงินได้ตรง, ไม่ถูกรถหรูไม่กี่คันดึงมากเท่า RMSE
- **Gating metric = MAPE ≤ 20%** บน test set — `GATE_MAPE = 0.20` ใน `src/register.py`
- **เชื่อมกับธุรกิจ** — MAE = 20.6% ของราคาเฉลี่ย test (896,128 INR); ทำนายคลาดไม่เกิน ±20% ได้ 63.2% ของรถ; แม่นสุดช่วง 700k–1.5M INR (MAPE 15.8%), แย่สุดรถ < 300k INR (MAPE 70.7%) — ตารางเต็มใน Canvas หัวข้อ 8

## หมวด 2: ข้อมูลและการตรวจสอบคุณภาพ (3 คะแนน)
- **แหล่งข้อมูล** — Kaggle `nehalbirla/vehicle-dataset-from-cardekho` (`Car details v3.csv`) ดาวน์โหลดด้วย `scripts/download_data.py` (kagglehub)
- **ขนาด** — 8,128 แถว → 6,926 แถว หลัง `drop_duplicates()` (ลบ 1,202) ใน `src/data_split.py`
- **การแบ่งชุด (ตามปี, `split_by_year`)** — train ปี ≤ 2016: 5,100 · val ปี 2017: 808 · test ปี ≥ 2018: 1,018
  - เหตุผล: เลียนแบบการใช้งานจริง (เทรนด้วยอดีต ทำนายอนาคต) และทำให้เห็น drift ตามเวลาได้จริง
  - ทำซ้ำได้: แบ่งด้วยเงื่อนไขปี ไม่มีการสุ่ม → ได้ชุดเดิมทุกครั้ง (ยืนยันแล้วว่ารัน pipeline ซ้ำใน venv ใหม่ได้ตัวเลขเท่าเดิมทุกตัว)
- **Schema (Pandera, `src/validation.py`)**
  - ช่วงค่า: `year` 1980–2026, `selling_price` 10,000–20,000,000, `km_driven` 0–3,000,000, `seats` 2–14
  - ค่าหมวดหมู่ที่อนุญาต: `fuel`, `seller_type`, `transmission`, `owner`
  - `strict=True` (ห้ามคอลัมน์แปลกปลอม), `lazy=True` (รายงานทุกข้อผิดพลาดพร้อมกัน)
- **สาธิตข้อมูลเสีย → ระบบหยุด + แจ้งเตือน**
  - ตอนเทรน: `load()` ใน `src/train.py` เรียก `validate()` ก่อนเทรน → ข้อมูลผิดโยน `SchemaErrors` → `src/pipeline.py` หยุดด้วย exit 1 และพิมพ์ `FAILED at: ...`
  - ตอนให้บริการ: API ตอบ **422** เมื่อ `fuel="Water"`, `year=2099`, `km_driven=-5` (`tests/test_api.py`)
  - unit test: `tests/test_validation.py` ข้อมูลดี 1 เคสผ่าน, ข้อมูลเสีย 4 เคสถูกจับ
  - **[TODO]** สคริปต์/คลิปสาธิต "ป้อนไฟล์ CSV เสียเข้า pipeline แล้วหยุด" สำหรับวันนำเสนอ
- **ป้องกัน Training-Serving Skew**
  - ฟังก์ชัน `clean_cars()` (`src/data_cleaning.py`) ตัวเดียว ถูกเรียกทั้งใน `src/train.py`, `src/api.py`, `src/monitor_drift.py`, `src/monitor_concept_drift.py`
  - imputer + scaler + one-hot + log-target ถูกห่อใน `TransformedTargetRegressor(Pipeline(...))` (`build()` ใน `src/train.py`) แล้ว export เป็นไฟล์เดียว `model_export/model.pkl` → API ใช้การแปลงชุดเดียวกับตอนเทรนแน่นอน
- **ค่าที่หายไป / ค่าผิดปกติ**
  - ค่าหาย: `SimpleImputer(strategy="median")` สำหรับตัวเลข; หมวดหมู่ที่ไม่รู้จักตอนเทรน → `OneHotEncoder(handle_unknown="ignore")` แต่ API กันไว้ก่อนด้วย `Literal`
  - ค่าผิดปกติ: ตัดด้วยช่วงค่าใน schema; **[TODO]** อธิบายเหตุผลที่ไม่ clip/IQR เพิ่ม (หรือทำเพิ่ม)

## หมวด 3: การพัฒนาโมเดล ประสิทธิภาพ และการอธิบายผล (3 คะแนน)
- **โมเดลที่เทียบ** (`MODELS` ใน `src/train.py`, ทุกตัวผ่าน `build()` เดียวกัน, เทรนบน log(ราคา))

  | โมเดล | hyperparameters หลัก | val MAE | val RMSE | val MAPE |
  |---|---|---|---|---|
  | baseline_median | strategy=median | 415,937 | 734,079 | 45.3% |
  | **ridge** | alpha=1.0 | **120,615** | **187,566** | 17.1% |
  | hist_gbm | learning_rate=0.1, max_iter=100, min_samples_leaf=20 | 129,150 | 356,127 | 14.4% |
  | random_forest | n_estimators=200, min_samples_leaf=2 | 135,776 | 363,139 | 15.0% |
- **เหตุผลเลือก Ridge** — MAE ต่ำสุด (optimizing metric) และ RMSE ต่ำกว่าโมเดลต้นไม้เกือบครึ่ง (ผิดพลาดหนักน้อยกว่า) + เล็ก เร็ว อธิบายง่าย
  - ต้องอธิบายข้อแลกเปลี่ยน: HistGBM/RF มี **MAPE ดีกว่า** (14.4%/15.0%) — เลือกตาม MAE ตามที่ตั้งไว้ตั้งแต่ต้น
- **ผลบน test (champion)** — MAE 184,906, MAPE 19.6%
- **[TODO]** การ tune hyperparameter (เช่น Ridge `alpha` หลายค่า) — ตอนนี้แต่ละโมเดลรัน 1 ค่า
- **[TODO]** การอธิบายผล: coefficient ของ Ridge / permutation importance, error analysis (มีแยกตามช่วงราคาแล้วใน Canvas)

## หมวด 4: การติดตามการทดลองและทะเบียนโมเดล (2 คะแนน)
- **MLflow tracking** (`sqlite:///mlflow.db`, experiment `used-car-price`) — บันทึก 6 อย่างใน `src/train.py`

  | สิ่งที่ต้องบันทึก | วิธี |
  |---|---|
  | เวอร์ชันโค้ด | param `git_sha` + tag อัตโนมัติ `mlflow.source.git.commit` |
  | เวอร์ชันข้อมูล | param `data_version` = MD5 ของ `train.csv` (10 ตัวแรก) |
  | ไฮเปอร์พารามิเตอร์ | `mlflow.log_params(est.get_params())` |
  | ตัวชี้วัด | `val_mae`, `val_rmse`, `val_mape` (+ `test_mae`, `test_mape` จาก `register.py`) |
  | ไฟล์ผลลัพธ์ | `mlflow.sklearn.log_model(...)` พร้อม signature |
  | สภาพแวดล้อม | params `python_version`, `sklearn_version`, `pandas_version`, `numpy_version` + `requirements.txt` ที่ล็อกทุกเวอร์ชัน |
  - ยืนยันด้วย `mlflow.search_runs()` แล้ว: 4 run ล่าสุด (git `8e2755c`) มี `data_version=1ea2f0cc54` (ตรงกับ MD5 ของ `train.csv`), `python_version=3.12.10`, `sklearn_version=1.9.1`, `pandas_version=3.0.6`, `numpy_version=2.5.3` ครบทุก run
  - run ชุดแรก 4 run (git `e4b3f27`) เทรนก่อนเพิ่มการ log ค่าเหล่านี้ → ใน UI ให้แคปเฉพาะ run ชุดล่าสุด
- **เปรียบเทียบข้ามการทดลอง** — MLflow UI หน้า Compare (4 runs) → แคปหน้าจอ
- **Model Registry** (`src/register.py`) — เลือก run MAE ต่ำสุด (ไม่รวม baseline) → register เป็นเวอร์ชันใหม่ของ `used-car-price` → ประเมินบน test → ผ่าน gate จึงตั้ง alias `champion`
- **Rollback** (`src/rollback.py`) — ย้าย alias `champion` กลับเวอร์ชันก่อนหน้า
  - registry ในเครื่องหลักตอนนี้มี version 1 และ 2 (`champion` = 2) — ทั้งสองเป็น Ridge ที่ได้ผลเท่ากัน เพราะเทรนจากข้อมูลและโค้ดเดียวกัน
  - ⚠️ `rollback.py` ยัง hardcode `target_version = "1"` → **[TODO] ข้อ 7**: หาเวอร์ชันก่อนหน้าอัตโนมัติ + `src/retrain.py` สาธิตวงจรเทรนใหม่

## หมวด 5: การให้บริการและโครงสร้างพื้นฐาน (3 คะแนน)
- **API (FastAPI, `src/api.py`)**
  - `POST /predict` → `predicted_price_inr`, `predicted_price_thb`, `model_version`
  - `GET /health` → `status`, `model_version`
  - `GET /metrics` → `total_requests`, `total_errors`, `p50_latency_ms`, `p95_latency_ms`
  - ตัวอย่างจริง (Maruti Swift Dzire VDI ปี 2014, 145,500 km): 438,475 INR / 171,005 THB
- **Container (`Dockerfile`)** — `python:3.12-slim`, ติดตั้ง `requirements-api.txt` (8 แพ็กเกจ ล็อกเวอร์ชัน), uvicorn 4 workers, port 8000, HEALTHCHECK ทุก 30 วินาทีเรียก `/health`
  - ต้อง build หลังรัน pipeline แล้ว (เพราะ `model_export/` ไม่อยู่ใน git)
- **Latency / Throughput / SLO** **[จากเอกสารเดิม]** (`scripts/load_test.py`, Docker 4 workers, พร้อมกัน 10)

  | SLO | เกณฑ์ | วัดได้ |
  |---|---|---|
  | p50 latency | ≤ 100 ms | 44.0 ms |
  | p95 latency | ≤ 250 ms | 86.3 ms |
  | Throughput | ≥ 50 req/s | 200.4 req/s |
  | Error rate | ≤ 1% | 0% |
  - ตรวจเทียบ SLO อัตโนมัติด้วย `src/check_system_health.py` (อ่าน `/health` + `/metrics`, exit 1 ถ้าเกิน)
- **รูปแบบการให้บริการ** — Real-time synchronous (เหตุผลใน `docs/SLO_AND_RETRAIN_POLICY.md`: ผู้ขายต้องการราคาทันที, โมเดลเล็ก inference เร็ว, โหลดไม่สูง)
- **Logs** — ทุกคำขอเขียน JSON (input, output, latency) ลง `logs/predictions.log`
- ข้อจำกัดที่ควรเขียน: `/metrics` เก็บในหน่วยความจำของแต่ละ worker (4 workers → ค่าต่อ worker), คำขอที่ตอบ 422 ไม่ถูกนับใน `total_errors`

## หมวด 6: การนำส่ง การเฝ้าระวัง และ CI/CD (3 คะแนน)
- **Data Drift** (`src/monitor_drift.py`, Evidently `DataDriftPreset`, train vs test) → `reports/drift_report.html`: **Dataset Drift detected, 11 จาก 12 คอลัมน์ (share 0.917)**
  - **[TODO]** โค้ดยังไม่เทียบเกณฑ์ 50% / ไม่ exit 1 เอง (เกณฑ์อยู่ในเอกสารเท่านั้น)
- **Concept Drift / คุณภาพการทำนาย** (`src/monitor_concept_drift.py`) — test MAE / val MAE = 184,906 / 120,615 = **1.53** > เกณฑ์ **1.3** → detected
- **สถานะระบบ** (`src/check_system_health.py`) — health + p95 ≤ 250 ms + error rate ≤ 1%
- **แยก Data vs Concept Drift** — data drift ดูการกระจายของ input (ไม่ต้องมี label), concept drift ดู error ที่เพิ่มขึ้นเมื่อมี label (ความสัมพันธ์ feature→ราคาเปลี่ยน)
- **นโยบายเทรนใหม่** (`docs/SLO_AND_RETRAIN_POLICY.md`) — trigger: data drift > 50% คอลัมน์, concept drift ratio > 1.3, หรือตรวจตามรอบทุกเดือน → เทรน → gate → promote / rollback
  - **[TODO] ข้อ 7**: สาธิตวงจรจริงด้วย `src/retrain.py`
- **CI (GitHub Actions, `.github/workflows/ci.yml`)** — 3 job แยกกัน, ติดตั้งจาก `requirements.txt` ที่ล็อกเวอร์ชัน

  | Job | ตรวจ | ผลจริง (run #26, PR #15) |
  |---|---|---|
  | 1. Code quality | `ruff check .` (ruff 0.16.9, กฎ E + F ใน `ruff.toml`) | ✅ All checks passed (12s) |
  | 2. Data validity | `tests/test_validation.py` + `tests/test_api.py` | ✅ 9/9 passed (1m 20s) |
  | 3. Model quality | `tests/test_model_quality.py` — เทรน Ridge บน fixture 1,274 แถว, MAPE ≤ 20% และชนะ baseline | ✅ 2/2 passed (1m 24s) |
  - fixture (`tests/fixtures/cars_sample.csv`): สุ่ม 1,500 แถว seed 42 เก็บปี ≤ 2017; Ridge MAPE 17.1% vs baseline 43.6%
- **หลักฐาน CI ไม่ผ่าน** — branch `demo/ci-failure` commit `0c5a5f9` แก้ช่วง `year` ใน schema เป็น 1980–2000 โดยตั้งใจ → test ล้ม → revert ใน `d381de7` → แคปหน้า Actions ทั้งสองครั้ง

## หมวด 7: การควบคุม Pipeline การทำซ้ำ และการกำกับดูแล (2 คะแนน)
- **Pipeline** (`src/pipeline.py`) — `data_split → validation → train → register → export_model` สั่งด้วย `python -m src.pipeline` คำสั่งเดียว, หยุดทันทีเมื่อขั้นใดล้ม, ใช้ `sys.executable` ให้ทุกขั้นรันใน Python/venv เดียวกัน
  - **[TODO]** เกณฑ์ต้องการ "Pipeline แบบ DAG" และตาราง Orchestration ไม่มีตัวเลือก "เขียนเอง" → พิจารณา Prefect/Dagster และเพิ่มขั้น download + build/run container
- **Reproducibility** — `requirements.txt` ล็อก 141 แพ็กเกจ (`pywin32` ติดตั้งเฉพาะ Windows), Python 3.12 ทั้ง Docker/CI, `random_state=42`, split ตามปี (ไม่สุ่ม)
  - ยืนยันแล้ว: venv ใหม่ + `pip install -r requirements.txt` + `python -m src.pipeline` → ได้ตัวเลขเท่าเดิมทุกตัว
- **Git / GitHub** — ทำงานผ่าน feature branch + Pull Request, merge ผ่าน Pull Request แล้ว 15 PR (#1–#16 ยกเว้น #4) + merge ตรง 1 ครั้ง (`9e26f8c`), รวม 44 commits
- **README** — **[TODO] ยังไม่มีขั้นตอนรันจากเครื่องเปล่า** (ตอนนี้มีแค่ภาพรวมและผลล่าสุด)
- **.gitignore** — ไม่นำ `data/`, `mlruns/`, `mlflow.db`, `model_export/`, `logs/`, `.env` ขึ้น repo

## หมวด 8: การนำเสนอ รายงาน และการทดสอบ (2 คะแนน)
- รายงาน: ทุกหัวข้อในเอกสารนี้ + Architecture Diagram (ส่วนท้าย)
- **เหตุผลการเลือกเครื่องมือ** (เอกสารโครงงานกำหนดให้อธิบาย)

  | หน้าที่ | เครื่องมือ | เหตุผลที่เลือก (ร่าง) |
  |---|---|---|
  | Version Control | Git + GitHub | branch/PR/review และเชื่อม GitHub Actions ได้ทันที |
  | Containerization | Docker | image `python:3.12-slim` เล็ก รันได้ทุกเครื่อง |
  | Data Validation | Pandera | ประกาศ schema เป็นโค้ด Python ใช้กับ pandas ตรงๆ, lazy validation |
  | Experiment Tracking | MLflow | บันทึก params/metrics/artifacts + UI เทียบ run |
  | Model Registry | MLflow Registry | มีเวอร์ชัน + alias (`champion`) ใช้ทำ gate/rollback |
  | Pipeline Orchestration | สคริปต์ `src/pipeline.py` | **[TODO]** ต้องให้เหตุผล หรือเปลี่ยนเป็น Prefect/Dagster |
  | Model Serving | FastAPI + uvicorn | validation ด้วย pydantic (ตอบ 422 อัตโนมัติ), เร็ว, เขียนง่าย |
  | Monitoring | Evidently + สคริปต์เขียนเอง | Evidently สำหรับ data drift; concept drift/system health เขียนเอง |
  | CI/CD | GitHub Actions | ฟรีสำหรับ repo, ผูกกับ PR |
- **เตรียม test case วันนำเสนอ** — ค่าหมวดหมู่ต้องตรงตัวพิมพ์ (เช่น `"First Owner"`); ส่งค่านอก schema ได้ 422 พร้อมข้อความบอกค่าที่รับได้
- **[TODO]** สไลด์ 12 นาที + แบ่งบทพูดให้สมาชิกทุกคน

## การใช้ AI ช่วยพัฒนา (บังคับตามเอกสารโครงงาน)
> ฉบับย่อของหัวข้อนี้อยู่ใน `README.md` ด้วย ข้อมูลด้านล่างมาจากประวัติ git — **[TODO] ทีมต้องเติมส่วนที่ใช้ AI ก่อนหน้านี้ที่ git ไม่ได้บันทึก**
- เครื่องมือ: Claude Code (Anthropic)
- ส่วนที่มีหลักฐานใน git (commit มี trailer `Co-Authored-By: Claude`):

  | Commit | งาน |
  |---|---|
  | `9e7b706` | ล็อกเวอร์ชัน `requirements.txt` + แก้ `pipeline.py` ให้ใช้ `sys.executable` |
  | `6117ed9` | CI 3 ด้าน, `ruff.toml`, `tests/test_model_quality.py`, fixture |
  | `8c44c83` | `Literal` ใน API + `tests/test_api.py` |
  | `c4e6d63` | ร่าง `docs/AI_PROJECT_CANVAS.md` |
  | (commit นี้) | ร่าง `docs/REPORT_OUTLINE.md` |
- งานที่ AI ช่วยตรวจ: เทียบ repo กับเกณฑ์การให้คะแนน, คำนวณตัวเลขธุรกิจจากโมเดลจริง
- ทีมรีวิวทุก diff และผลทดสอบก่อน merge; ต้องอธิบายโค้ดทุกบรรทัดได้ตามข้อกำหนด
- หมายเหตุสภาพแวดล้อม: บาง commit เดิมรันบน Google Colab เพราะเครื่อง local ติด DLL block (`79ac97e`, `e7faeb4`)

---

## Architecture Diagram — คำอธิบายสำหรับวาดใน Canva

### แนะนำเลย์เอาต์
แบ่งเป็น 4 แถบแนวนอน (swimlane) จากบนลงล่าง ลูกศรหลักไหลซ้าย → ขวา:
1. **Data & Training (offline)** 2. **Registry & Release** 3. **Serving (online, ใน Docker)** 4. **Monitoring & CI/CD**

### Components
| ID | Component | ไฟล์ / เครื่องมือ | Input → Output |
|---|---|---|---|
| **A** | Data Source | Kaggle CarDekho, `scripts/download_data.py` | → `data/raw/car_details_v3.csv` (8,128 แถว) |
| **B** | Split + Dedupe | `src/data_split.py` | raw → `data/processed/{train,val,test}.csv` (5,100 / 808 / 1,018) |
| **C** | Validation | `src/validation.py` (Pandera) | CSV → ผ่าน หรือ `SchemaErrors` (pipeline หยุด) |
| **D** | Cleaning (shared) | `src/data_cleaning.py` `clean_cars()` | ข้อความ เช่น "23.4 kmpl" → ตัวเลข, `name` → `brand` |
| **E** | Training | `src/train.py` — 4 โมเดล ผ่าน `build()` | clean data → โมเดล + metrics |
| **F** | Experiment Tracking | MLflow (`mlflow.db`, `mlruns/`) | params, metrics, artifacts, git SHA, data version, env |
| **G** | Model Registry + Gate | `src/register.py` (MLflow Registry) | best run → version ใหม่ → test MAPE ≤ 20%? → alias `champion` |
| **H** | Rollback | `src/rollback.py` | ย้าย alias `champion` กลับเวอร์ชันก่อน |
| **I** | Export | `src/export_model.py` | `champion` → `model_export/model.pkl` |
| **J** | Pipeline Runner | `src/pipeline.py` | สั่ง B → C → E → G → I ตามลำดับ |
| **K** | Docker Container | `Dockerfile` (python:3.12-slim, 4 workers, HEALTHCHECK) | บรรจุ L + D + `model.pkl` |
| **L** | FastAPI Service | `src/api.py` | `/predict`, `/health`, `/metrics` |
| **M** | Client | ผู้ขาย / `curl` / `scripts/load_test.py` | JSON สเปกรถ → ราคา INR + THB |
| **N** | Prediction Logs | `logs/predictions.log` | input, output, latency ต่อคำขอ |
| **O** | Data Drift Monitor | `src/monitor_drift.py` (Evidently) | train vs ข้อมูลใหม่ → `reports/drift_report.html` |
| **P** | Concept Drift Monitor | `src/monitor_concept_drift.py` | MAE ใหม่ / MAE val > 1.3 → alert |
| **Q** | System Health Check | `src/check_system_health.py` | `/health` + `/metrics` เทียบ SLO → HEALTHY / UNHEALTHY |
| **R** | Retrain Policy | `docs/SLO_AND_RETRAIN_POLICY.md` | alert จาก O/P → สั่ง J ใหม่ |
| **S** | GitHub (branch + PR) | GitHub | push / pull request → trigger T |
| **T** | CI | GitHub Actions `ci.yml` | ruff · data/API tests · model quality gate |

### Data flow (ลูกศร)
1. **A → B** ดาวน์โหลดข้อมูลดิบ แล้ว dedupe + แบ่งตามปี
2. **B → C** ตรวจ schema ทุกชุด ❌ ไม่ผ่าน → pipeline หยุด (exit 1)
3. **C → D → E** clean ด้วยฟังก์ชันร่วม แล้วเทรน 4 โมเดล
4. **E → F** log ครบ 6 อย่างของทุก run
5. **F → G** เลือก run MAE ต่ำสุด → register → ประเมิน test → ผ่าน gate → `champion`
6. **G → I** export `champion` เป็น `model.pkl`
7. **I → K** `docker build` คัดลอก `model.pkl` + `api.py` + `data_cleaning.py` เข้า image
8. **M → L → D → model → M** คำขอ `/predict` → pydantic ตรวจ (ผิด → 422) → `clean_cars()` ตัวเดียวกับตอนเทรน → ทำนาย → ตอบ INR + THB
9. **L → N** เขียน log ทุกคำขอ; **L → Q** `/metrics` ให้ health check อ่าน
10. **B/N → O, P** ข้อมูลใหม่ (จำลองด้วย test set ปี ≥ 2018) → ตรวจ drift
11. **O, P → R → J** เกินเกณฑ์ → เทรนใหม่ทั้ง pipeline → gate ใหม่
12. **G ⇄ H** ถ้าเวอร์ชันใหม่แย่ → rollback alias `champion`
13. **S → T** ทุก PR เข้า `main` → CI 3 job ต้องผ่านก่อน merge

### เส้นประ / สีที่แนะนำ
- เส้นทึบ = data flow หลัก, เส้นประ = สัญญาณควบคุม/alert (O, P, Q → R; H → G)
- กล่องสีแดงเล็กที่ C และ L = "จุดที่ปฏิเสธข้อมูลเสีย"
- ไฮไลต์ D ให้เห็นว่าอยู่ทั้งฝั่ง training และ serving (จุดป้องกัน training-serving skew)
