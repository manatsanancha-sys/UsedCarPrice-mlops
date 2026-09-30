import subprocess
import hashlib
import sklearn

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_cleaning import clean_cars
from src.validation import validate

NUM = ["year", "km_driven", "mileage", "engine", "max_power", "torque", "seats"]
CAT = ["brand", "fuel", "seller_type", "transmission", "owner"]
TARGET = "selling_price"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("used-car-price")


def load(name):
    raw = pd.read_csv(f"data/processed/{name}.csv")
    validate(raw)  # ข้อมูลผิดปกติ -> หยุดทันที
    df = clean_cars(raw)
    return df[NUM + CAT], df[TARGET]


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"




def data_version(path="data/processed/train.csv"):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:10]

def build(estimator):
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])
    pipe = Pipeline([("pre", pre), ("model", estimator)])
    # เทรนบน log(ราคา) แต่ผลที่ทำนายออกมาเป็นราคาจริง (รูปี)
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


MODELS = {
    "baseline_median": DummyRegressor(strategy="median"),
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42
    ),
    "hist_gbm": HistGradientBoostingRegressor(random_state=42),
}


def log_model(model, signature):
    try:
        mlflow.sklearn.log_model(model, name="model", signature=signature, serialization_format="cloudpickle")
    except TypeError:  # MLflow รุ่นเก่า
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature, serialization_format="cloudpickle")


def main():
    X_tr, y_tr = load("train")
    X_va, y_va = load("val")
    results = []

    for name, est in MODELS.items():
        with mlflow.start_run(run_name=name):
            model = build(est)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)

            metrics = {
                "val_mae": mean_absolute_error(y_va, pred),
                "val_rmse": root_mean_squared_error(y_va, pred),
                "val_mape": mean_absolute_percentage_error(y_va, pred),
            }
            mlflow.log_params(est.get_params())
            mlflow.log_params(
                {
                    "model_name": name,
                    "train_rows": len(X_tr),
                    "val_rows": len(X_va),
                    "git_sha": git_sha(),
                    "data_version": data_version(),
                    "python_version": __import__("sys").version.split()[0],
                    "sklearn_version": sklearn.__version__,
                    "pandas_version": pd.__version__,
                    "numpy_version": np.__version__,
                }
            )
            mlflow.log_metrics(metrics)
            log_model(model, infer_signature(X_va, pred))
            results.append((name, metrics))

    print("\n=== validation results (ราคาเป็นรูปี) ===")
    for name, m in sorted(results, key=lambda r: r[1]["val_mae"]):
        print(f"{name:16s} MAE={m['val_mae']:>10,.0f}  RMSE={m['val_rmse']:>10,.0f}  MAPE={m['val_mape']:.1%}")


if __name__ == "__main__":
    main()

