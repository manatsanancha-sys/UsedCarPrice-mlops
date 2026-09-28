import pandas as pd
from pathlib import Path

RAW = Path("data/raw/car_details_v3.csv")
OUT = Path("data/processed")
TRAIN_MAX_YEAR = 2016   # train: year <= 2016
VAL_YEAR = 2017         # val: year == 2017, test: year > 2017


def split_by_year(df: pd.DataFrame):
    train = df[df["year"] <= TRAIN_MAX_YEAR]
    val = df[df["year"] == VAL_YEAR]
    test = df[df["year"] > VAL_YEAR]
    return train, val, test


def main():
    df = pd.read_csv(RAW)
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"rows: {n0} -> {len(df)} after dedupe (removed {n0 - len(df)})")

    train, val, test = split_by_year(df)
    print("year range:", df["year"].min(), "-", df["year"].max())
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False)
    val.to_csv(OUT / "val.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
