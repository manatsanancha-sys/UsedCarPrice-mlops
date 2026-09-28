import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.train import load

NAME = "used-car-price"
GATE_MAPE = 0.20  # ด่านตรวจ: MAPE บน test ต้องไม่เกิน 20% (ปรับได้ ให้ตรงกับ Canvas)


def main():
    client = MlflowClient()
    runs = mlflow.search_runs(
        experiment_names=["used-car-price"],
        filter_string="params.model_name != 'baseline_median'",
        order_by=["metrics.val_mae ASC"],
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
    print(f"TEST  MAE={mae:,.0f}  MAPE={mape:.1%}")

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics({"test_mae": mae, "test_mape": mape})

    if mape <= GATE_MAPE:
        client.set_registered_model_alias(NAME, "champion", mv.version)
        print(f"PASS gate -> version {mv.version} promoted to alias 'champion'")
    else:
        print(f"FAIL gate (MAPE {mape:.1%} > {GATE_MAPE:.0%}) -> NOT promoted")


if __name__ == "__main__":
    main()
