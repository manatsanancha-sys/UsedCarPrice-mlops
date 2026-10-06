import re
import numpy as np
import pandas as pd

REF_YEAR = 2020
BRAND_FIX = {"Land": "Land Rover", "Ashok": "Ashok Leyland"}


def _first_number(s):
    """ดึงตัวเลขตัวแรกจากข้อความ เช่น '23.4 kmpl' -> 23.4"""
    if pd.isna(s):
        return np.nan
    m = re.search(r"\d+(?:[.,]\d+)*", str(s))
    if not m:
        return np.nan
    return float(m.group().replace(",", ""))


def _torque_nm(s):
    """แปลง torque ให้เป็น Nm; ค่าที่เป็น kgm จริงๆ จะเป็นเลขน้อย (< 50)"""
    v = _first_number(s)
    if pd.isna(v):
        return np.nan
    text = str(s).lower()
    if re.match(r"\s*\d+(?:[.,]\d+)?\s*nm", text):
        return v
    if re.match(r"\s*\d+(?:[.,]\d+)?\s*kgm", text):
        return v * 9.80665
    return v * 9.80665 if v < 50 else v


def clean_cars(df: pd.DataFrame) -> pd.DataFrame:
    """ฟังก์ชัน clean เดียว ใช้ร่วมกันทั้ง training และ serving"""
    df = df.copy()
    brand = df["name"].astype(str).str.split().str[0]
    df["brand"] = brand.replace(BRAND_FIX)
    for c in ["mileage", "engine", "max_power"]:
        df[c] = df[c].map(_first_number)
    df["torque"] = df["torque"].map(_torque_nm)
    return df.drop(columns=["name"])