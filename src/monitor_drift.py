import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.data_cleaning import clean_cars

NUM = ["year", "km_driven", "mileage", "engine", "max_power", "torque", "seats"]
CAT = ["brand", "fuel", "seller_type", "transmission", "owner"]


def main():
    ref = clean_cars(pd.read_csv("data/processed/train.csv"))[NUM + CAT]
    cur = clean_cars(pd.read_csv("data/processed/test.csv"))[NUM + CAT]

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=ref, current_data=cur)
    result.save_html("reports/drift_report.html")
    print("saved reports/drift_report.html")

    d = result.dict()
    print(d)


if __name__ == "__main__":
    main()
