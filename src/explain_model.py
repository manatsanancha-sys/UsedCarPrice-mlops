"""อธิบายผลโมเดล champion (Ridge): coefficient ของแต่ละ feature เรียงตามขนาดผลต่อราคา

โหลด model_export/model.pkl (champion ที่ export ไว้ = ตัวเดียวกับที่ API ใช้) แล้วเช็คว่าตรงกับ alias champion
โมเดลเทรนบน log(ราคา) และ feature ตัวเลขผ่าน StandardScaler ดังนั้น
  - feature ตัวเลข: coef = ผลต่อ log(ราคา) เมื่อค่าเพิ่ม 1 SD -> ราคาเปลี่ยน (e^coef - 1) %
  - feature หมวดหมู่ (one-hot): coef = ผลของการเป็นหมวดนั้น เทียบกับค่าเฉลี่ยของหมวดอื่นในโมเดล
ผลบันทึกเป็น docs/model_explanation.md
"""
from pathlib import Path

import cloudpickle
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.train import CAT, NUM

NAME = "used-car-price"
OUT = Path("docs/model_explanation.md")
TOP_N = 20

# เหตุผลเชิงโดเมนสำหรับ feature ที่มักมีผลต่อราคารถมือสอง (ใช้ประกอบคำอธิบายในไฟล์ .md)
REASONS = {
    "year": "รถใหม่กว่าเสื่อมราคาน้อยกว่า — ค่าเสื่อมตามอายุเป็นปัจจัยหลักของราคารถมือสอง",
    "max_power": "แรงม้าสูงสะท้อนรุ่น/ระดับรถที่แพงกว่า (รถสเปกสูง รถหรู)",
    "engine": "ขนาดเครื่องยนต์ใหญ่มักมากับรถขนาดใหญ่/ราคาตั้งต้นสูง",
    "km_driven": "เลขไมล์มากแปลว่าสึกหรอมาก ราคาลดลง",
    "torque": "แรงบิดสูงสัมพันธ์กับเครื่องดีเซล/รถ SUV ที่ราคาสูง",
    "mileage": "อัตราสิ้นเปลืองดี (กม./ลิตรสูง) มักเป็นรถเล็กประหยัด ราคาต่ำกว่า",
    "seats": "จำนวนที่นั่งบอกประเภทรถ (รถเก๋ง/ MPV / SUV)",
    "brand": "แบรนด์กำหนดราคาตั้งต้นและการรักษามูลค่า (แบรนด์หรูราคาสูงกว่ามาก)",
    "transmission": "เกียร์อัตโนมัติพบในรถรุ่นสูงกว่า ราคาจึงสูงกว่า",
    "fuel": "ดีเซลมักเป็นรถใหญ่/ราคาสูงกว่า, CNG/LPG มักเป็นรถเล็กราคาต่ำ",
    "seller_type": "ดีลเลอร์ตั้งราคาสูงกว่าผู้ขายรายบุคคล (มีการรับประกัน/ปรับสภาพรถ)",
    "owner": "ผ่านมือหลายเจ้าของ ความน่าเชื่อถือลดลง ราคาลดลง",
}


def readable(feature_name):
    """'num__year' -> ('year', 'year'); 'cat__brand_Maruti' -> ('brand', 'brand = Maruti')"""
    kind, rest = feature_name.split("__", 1)
    if kind == "num":
        return rest, rest
    for col in CAT:
        if rest.startswith(col + "_"):
            return col, f"{col} = {rest[len(col) + 1:]}"
    return rest, rest


def coefficient_table(model):
    pipe = model.regressor_
    pre, ridge = pipe.named_steps["pre"], pipe.named_steps["model"]
    names = pre.get_feature_names_out()
    scaler = pre.named_transformers_["num"].named_steps["sc"]
    sd = dict(zip(NUM, scaler.scale_))

    rows = []
    for fname, coef in zip(names, ridge.coef_):
        feature, label = readable(fname)
        rows.append({
            "feature": feature,
            "label": label,
            "coef": coef,
            "effect_pct": (np.exp(coef) - 1) * 100,
            "unit": f"ต่อ +1 SD ({sd[feature]:,.1f})" if feature in sd else "เมื่อเป็นหมวดนี้",
        })
    df = pd.DataFrame(rows)
    df["abs_coef"] = df["coef"].abs()
    return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def feature_importance(df):
    """รวมเป็นรายคอลัมน์ต้นฉบับ: ตัวเลข = |coef|, หมวดหมู่ = ช่วง coef (สูงสุด - ต่ำสุด) ระหว่างหมวด"""
    rows = []
    for feature, g in df.groupby("feature"):
        spread = g["coef"].abs().iloc[0] if feature in NUM else g["coef"].max() - g["coef"].min()
        rows.append({"feature": feature, "type": "ตัวเลข" if feature in NUM else "หมวดหมู่", "importance": spread})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    champ = MlflowClient().get_model_version_by_alias(NAME, "champion").version
    with open("model_export/model.pkl", "rb") as f:
        exported = cloudpickle.load(f)
    version = exported["version"]
    note = ""
    if str(version) != str(champ):
        note = f" (คำเตือน: registry champion = v{champ} ให้รัน python -m src.export_model ก่อน)"
    print(f"model_export/model.pkl = version {version}, registry champion = version {champ}{note}\n")

    coefs = coefficient_table(exported["model"])
    importance = feature_importance(coefs)

    print(f"=== Top {TOP_N} coefficients (เรียงตาม |coef|) ===")
    print(f"{'#':>2}  {'feature':<36} {'coef':>8} {'ผลต่อราคา':>10}  หน่วย")
    for i, r in coefs.head(TOP_N).iterrows():
        print(f"{i + 1:>2}  {r.label:<36} {r.coef:>8.3f} {r.effect_pct:>+9.1f}%  {r.unit}")

    print("\n=== ความสำคัญรายคอลัมน์ต้นฉบับ ===")
    for i, r in importance.iterrows():
        print(f"{i + 1:>2}  {r.feature:<14} {r.type:<9} {r.importance:.3f}")

    lines = [
        "# คำอธิบายผลโมเดล (Model Explanation)",
        "",
        f"> สร้างโดย `python -m src.explain_model` จาก `model_export/model.pkl` (champion version {version}, Ridge)",
        "> โมเดลทำนาย log(ราคา) และ feature ตัวเลขถูก standardize — ผลต่อราคา = (e^coef − 1) %",
        "> ตัวเลข: ผลเมื่อค่าเพิ่ม 1 SD (ค่า SD ในวงเล็บ) · หมวดหมู่: ผลของการเป็นหมวดนั้นเทียบกับหมวดอื่นในโมเดล",
        "",
        "## ความสำคัญรายคอลัมน์ต้นฉบับ",
        "",
        "ตัวเลข = |coef| · หมวดหมู่ = ช่วงห่างของ coef ระหว่างหมวดที่สูงสุดกับต่ำสุด",
        "",
        "| # | feature | ประเภท | ความสำคัญ | เหตุผลเชิงโดเมน |",
        "|---|---|---|---|---|",
    ]
    for i, r in importance.iterrows():
        lines.append(f"| {i + 1} | `{r.feature}` | {r.type} | {r.importance:.3f} | {REASONS.get(r.feature, '')} |")
    lines += [
        "",
        f"## Top {TOP_N} coefficients",
        "",
        "| # | feature | coef | ผลต่อราคา | หน่วย |",
        "|---|---|---|---|---|",
    ]
    for i, r in coefs.head(TOP_N).iterrows():
        lines.append(f"| {i + 1} | {r.label} | {r.coef:.3f} | {r.effect_pct:+.1f}% | {r.unit} |")
    numeric = coefs[coefs["feature"].isin(NUM)]
    lines += [
        "",
        "## Coefficient ของ feature ตัวเลขทั้งหมด",
        "",
        "| feature | coef | ผลต่อราคา | หน่วย |",
        "|---|---|---|---|",
    ]
    for _, r in numeric.iterrows():
        lines.append(f"| {r.label} | {r.coef:.3f} | {r.effect_pct:+.1f}% | {r.unit} |")
    lines += [
        "",
        "## การเลือก alpha: เลือกจาก val ไม่ใช่ test (ป้องกัน data leakage)",
        "",
        "Ridge ลอง alpha 5 ค่าใน `src/train.py` (`RIDGE_ALPHAS`) — เทรนบน train (ปี ≤ 2016)"
        " แล้วเลือกจาก **val MAE** (ปี 2017)",
        "",
        "| alpha | val MAE | test MAE |",
        "|---|---|---|",
        "| 0.1 | 121,908 | **182,880** ← ดีที่สุดถ้าดู test |",
        "| **1.0** | **120,615** ← เลือก | 184,906 |",
        "| 10.0 | 124,064 | 192,207 |",
        "| 50.0 | 126,131 | 198,244 |",
        "| 100.0 | 127,124 | 200,219 |",
        "",
        "- เลือก **alpha = 1.0** เพราะ val MAE ต่ำสุด → test MAE 184,906",
        "- ถ้าเลือกจาก test จะได้ alpha = 0.1 (test MAE 182,880 ดีกว่า 2,026) **แต่ไม่ทำ** เพราะ test ต้องเป็นข้อมูลที่",
        "  ไม่ถูกใช้ตัดสินใจใดๆ ก่อนประเมินผลสุดท้าย — ถ้าใช้ test เลือก hyperparameter ตัวเลข test จะดีเกินจริง (data leakage)",
        "  และไม่บอกว่าโมเดลจะทำงานกับข้อมูลใหม่ได้ดีแค่ไหน",
        "- ตัวเลขตารางนี้วัดจาก `python -m src.pipeline` (ทุกค่าบันทึกเป็น run `ridge_alpha=…` ใน MLflow;"
        " run ที่ถูกเลือกมี tag `best_ridge_alpha=true`)",
        "",
        "## ข้อควรระวังในการตีความ",
        "",
        "- coefficient ของ Ridge บอก **ความสัมพันธ์** ในข้อมูลเทรน ไม่ใช่เหตุและผล",
        "- feature ที่สัมพันธ์กันสูง (เช่น engine, max_power, torque) แบ่งผลกันเอง ค่าแต่ละตัวจึงอาจดูเล็กหรือกลับทิศ",
        "- แบรนด์ที่มีข้อมูลน้อยอาจได้ coefficient สุดโต่ง — Ridge (alpha) ช่วยหดค่าเหล่านี้ลงแล้วบางส่วน",
        "- **`brand = Land` หมายถึง Land Rover** — `clean_cars()` ใช้คำแรกของชื่อรุ่นเป็นยี่ห้อ ยี่ห้อที่ชื่อมีหลายคำ",
        "  จึงเหลือแค่คำแรก (ข้อจำกัดของการ clean ข้อมูล; ไม่กระทบการทำนายเพราะใช้วิธีเดียวกันทั้งตอนเทรนและให้บริการ)",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
