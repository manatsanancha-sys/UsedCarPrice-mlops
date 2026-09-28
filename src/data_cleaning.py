import re
import numpy as np
import pandas as pd


def _first_number(s):
    """ดึงตัวเลขตัวแรกจากข้อความ เช่น '23.4 kmpl' -> 23.4"""
    if pd.isna(s):
        return np.nan
    m = re.search(r"\d+(?:[.,]\d+)*", str(s))
    if not m:
        return np.nan
    return float(m.group().replace(",", ""))


def _torque_nm(s):
    """แปลง torque ให้เป็น Nm (ถ้าเป็น kgm คูณ 9.80665)"""
    v = _first_number(s)
    if pd.isna(v):
        return np.nan
    return v * 9.80665 if "kgm" in str(s).lower() else v


def clean_cars(df: pd.DataFrame) -> pd.DataFrame:
    """ฟังก์ชัน clean เดียว ใช้ร่วมกันทั้ง training และ serving"""
    df = df.copy()
    df["brand"] = df["name"].astype(str).str.split().str[0]
    for c in ["mileage", "engine", "max_power"]:
        df[c] = df[c].map(_first_number)
    df["torque"] = df["torque"].map(_torque_nm)
    return df.drop(columns=["name"])
