"""สาธิตวงจรเทรนใหม่ (retrain): เทรน challenger ด้วยข้อมูลใหม่กว่า -> register -> เทียบกับ champion
บน test set เดียวกัน -> promote เฉพาะเมื่อผ่าน gate และไม่แย่กว่า champion

จำลอง "ได้ข้อมูลใหม่เข้ามา" โดยเทรนด้วย train + val (ปี <= 2017) แทน train อย่างเดียว (ปี <= 2016)
ใช้ --alpha เพื่อสร้าง challenger ที่แย่กว่าโดยตั้งใจ (สาธิตกรณีไม่ promote) เช่น --alpha 1000
"""
import argparse
import hashlib
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.register import GATE_MAPE, NAME, REF_YEAR, reference_mae
from src.train import MODELS, build, git_sha, load, log_model


def data_version(names):
    h = hashlib.md5()
    for name in names:
        with open(f"data/processed/{name}.csv", "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:10]


def evaluate(model, X, y):
    pred = model.predict(X)
    return mean_absolute_error(y, pred), mean_absolute_percentage_error(y, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=MODELS["ridge"].alpha)
    args = parser.parse_args()

    client = MlflowClient()
    champ = client.get_model_version_by_alias(NAME, "champion")
    print(f"champion ปัจจุบัน: version {champ.version}")

    # 1) เทรน challenger ด้วยข้อมูลที่ใหม่กว่า
    X_tr, y_tr = load("train")
    X_va, y_va = load("val")
    X_new, y_new = pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va])
    X_te, y_te = load("test")

    with mlflow.start_run(run_name="ridge_retrain") as run:
        est = Ridge(alpha=args.alpha)
        model = build(est).fit(X_new, y_new)
        mae, mape = evaluate(model, X_te, y_te)
        ref = reference_mae(model, X_te, y_te)
        mlflow.log_params(est.get_params())
        mlflow.log_params(
            {
                "model_name": "ridge_retrain",
                "train_data": "train+val",
                "train_rows": len(X_new),
                "git_sha": git_sha(),
                "data_version": data_version(["train", "val"]),
                "python_version": sys.version.split()[0],
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
                "ref_mae": round(ref, 2),
            }
        )
        mlflow.log_metrics({"test_mae": mae, "test_mape": mape})
        log_model(model, infer_signature(X_te, model.predict(X_te)))

    # 2) register เป็นเวอร์ชันใหม่
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", NAME)
    print(f"registered challenger: version {mv.version} (alpha={args.alpha}, train+val {len(X_new)} แถว)")

    # 3) เทียบกับ champion บน test set เดียวกัน
    champ_model = mlflow.sklearn.load_model(f"models:/{NAME}/{champ.version}")
    champ_mae, champ_mape = evaluate(champ_model, X_te, y_te)
    print("\n=== เทียบบน test set เดียวกัน ===")
    print(f"champion   v{champ.version}: MAE={champ_mae:>10,.0f}  MAPE={champ_mape:.1%}")
    print(f"challenger v{mv.version}: MAE={mae:>10,.0f}  MAPE={mape:.1%}  ref_mae({REF_YEAR})={ref:,.0f}")

    # 4) ตัดสินใจ: ต้องผ่าน gate และ MAE ไม่แย่กว่า champion
    passed_gate = mape <= GATE_MAPE
    not_worse = mae <= champ_mae
    if passed_gate and not_worse:
        client.set_registered_model_alias(NAME, "champion", mv.version)
        change = f"MAE ดีขึ้น {champ_mae - mae:,.0f}" if mae < champ_mae else "MAE ไม่แย่กว่าเดิม"
        print(f"\nPROMOTE -> champion = version {mv.version} (MAPE {mape:.1%} <= {GATE_MAPE:.0%} และ {change})")
        print("ขั้นถัดไป: python -m src.export_model แล้ว build Docker image ใหม่")
    else:
        reasons = []
        if not passed_gate:
            reasons.append(f"MAPE {mape:.1%} เกินเกณฑ์ {GATE_MAPE:.0%}")
        if not not_worse:
            reasons.append(f"MAE แย่กว่า champion {mae - champ_mae:,.0f}")
        print(f"\nNOT PROMOTED -> คง champion = version {champ.version} ({'; '.join(reasons)})")


if __name__ == "__main__":
    main()
