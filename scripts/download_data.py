import kagglehub, shutil, pathlib

path = kagglehub.dataset_download("nehalbirla/vehicle-dataset-from-cardekho")
dst = pathlib.Path("data/raw")
dst.mkdir(parents=True, exist_ok=True)
shutil.copy(pathlib.Path(path) / "Car details v3.csv", dst / "car_details_v3.csv")
print("saved to", dst)
