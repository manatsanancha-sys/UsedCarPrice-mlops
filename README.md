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

## การใช้ AI ช่วยพัฒนา

ตามข้อกำหนดรายวิชา ระบุส่วนที่ใช้ AI ช่วย (เครื่องมือ: **Claude Code**, Anthropic) ทุกส่วนผ่านการรีวิว diff
และผลทดสอบโดยทีมก่อน merge และสมาชิกต้องอธิบายโค้ดทุกบรรทัดได้

| Commit | งานที่ AI ช่วย |
|---|---|
| `9e7b706` | ล็อกเวอร์ชัน `requirements.txt` + แก้ `src/pipeline.py` ให้ใช้ `sys.executable` |
| `6117ed9` | CI 3 ด้าน (ruff / data validity / model quality), `ruff.toml`, `tests/test_model_quality.py` + fixture |
| `8c44c83` | ใช้ `Literal` ปฏิเสธค่าหมวดหมู่ผิดใน `src/api.py` + `tests/test_api.py` |
| `c4e6d63` | ร่าง `docs/AI_PROJECT_CANVAS.md` |
| — | ร่าง `docs/REPORT_OUTLINE.md` |

- AI ช่วยตรวจ repo เทียบกับเกณฑ์การให้คะแนน และคำนวณตัวเลขเชิงธุรกิจจากโมเดลจริง
- commit ที่ AI ช่วยมีบรรทัด `Co-Authored-By: Claude` ตรวจได้ด้วย `git log --grep="Co-Authored-By: Claude"`
- **[TODO ทีม]** เติมส่วนที่ใช้ AI ก่อนหน้านี้ที่ไม่ได้บันทึกใน git
