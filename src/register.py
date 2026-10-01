import sys

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.train import load

NAME = "used-car-price"
GATE_MAPE = 0.20  # ด่านตรวจ: MAPE บน test ต้องไม่เกิน 20% (ปรับได้ ให้ตรงกับ Canvas)
REF_YEAR = 2018  # ปีแรกของ test = ช่วงอ้างอิงตอน deploy; ปีหลังจากนี้ใช้ตรวจ concept drift


def reference_mae(model, X, y):
    """MAE บน test เฉพาะปี REF_YEAR -> บันทึกเป็น param ref_mae ให้ monitor_concept_drift ใช้"""
    mask = X["year"] == REF_YEAR
    return mean_absolute_error(y[mask], model.predict(X[mask]))


def main():
    client = MlflowClient()
    runs = mlflow.search_runs(
        experiment_names=["used-car-price"],
        filter_string="params.model_name != 'baseline_median'",
        order_by=["metrics.val_mae ASC", "attributes.start_time DESC"],  # เสมอกัน -> เอา run ล่าสุด
        max_results=1,
    )
    best = runs.iloc[0]
    print("best run:", best["params.model_name"], "val_mae =", round(best["metrics.val_mae"]))

    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", NAME)
    print("registered version:", mv.version)

    model = mlflow.sklearn.load_model(f"models:/{NAME}/{mv.version}")
    X_te, y_te = load("test")
    pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    mape = mean_absolute_percentage_error(y_te, pred)
    ref = reference_mae(model, X_te, y_te)
    print(f"TEST  MAE={mae:,.0f}  MAPE={mape:.1%}  ref_mae({REF_YEAR})={ref:,.0f}")

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics({"test_mae": mae, "test_mape": mape})
        mlflow.log_param("ref_mae", round(ref, 2))

    if mape <= GATE_MAPE:
        client.set_registered_model_alias(NAME, "champion", mv.version)
        print(f"PASS gate -> version {mv.version} promoted to alias 'champion'")
    else:
        print(f"FAIL gate (MAPE {mape:.1%} > {GATE_MAPE:.0%}) -> NOT promoted")
        sys.exit(1)  # หยุด pipeline ไม่ให้ export โมเดลที่ไม่ผ่านด่าน


if __name__ == "__main__":
    main()
