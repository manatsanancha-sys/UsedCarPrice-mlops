from pathlib import Path

import cloudpickle
import mlflow.sklearn
from mlflow import MlflowClient

from src.train import CAT, NUM

NAME = "used-car-price"


def main():
    version = MlflowClient().get_model_version_by_alias(NAME, "champion").version
    model = mlflow.sklearn.load_model(f"models:/{NAME}@champion")
    Path("model_export").mkdir(exist_ok=True)
    with open("model_export/model.pkl", "wb") as f:
        cloudpickle.dump(
            {"model": model, "features": NUM + CAT, "num": NUM, "version": str(version)}, f
        )
    print("exported champion version", version)


if __name__ == "__main__":
    main()
