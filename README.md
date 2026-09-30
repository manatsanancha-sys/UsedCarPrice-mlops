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
