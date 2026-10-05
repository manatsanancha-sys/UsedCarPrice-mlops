# static/ — ไฟล์สำหรับหน้า /docs แบบออฟไลน์

หน้า Swagger UI (`/docs`) โหลดไฟล์จากโฟลเดอร์นี้แทน CDN จึงเปิดได้แม้ไม่มีอินเทอร์เน็ต
(ตามแนวทาง "Self-hosting JavaScript and CSS for docs" ของ FastAPI)

| ไฟล์ | ที่มา |
|---|---|
| `swagger-ui-bundle.js`, `swagger-ui.css` | npm `swagger-ui-dist@5.33.1` (https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.33.1/) |
| `swagger-ui-LICENSE`, `swagger-ui-NOTICE`, `swagger-ui-swagger-ui-bundle.js.LICENSE.txt` | license ของ Swagger UI (Apache-2.0) จากแพ็กเกจเดียวกัน |
| `index.html` | หน้า UI ภาษาไทยที่ `/` และ `/app` (เขียนเอง ไฟล์เดียว ไม่ใช้ CDN/ฟอนต์ภายนอก) |
| `favicon.png` | https://fastapi.tiangolo.com/img/favicon.png (favicon เริ่มต้นของ FastAPI) |

อัปเดต Swagger UI: ดาวน์โหลดไฟล์ทั้งสองจากเวอร์ชันใหม่ของ `swagger-ui-dist` มาแทน แล้วแก้เวอร์ชันในตารางนี้
