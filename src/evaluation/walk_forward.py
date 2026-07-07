import os
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from src.config import MODEL_DATASET, TARGETS, TEST_DAYS, RANDOM_STATE


EXCLUDE = [
    "date",
    "source_file",
    "sheet",
    "holiday_name",
    "total",
    "delivery",
    "digital",
    "cash",
    "delivery_share",
    "digital_share",
    "cash_share",
    "pos",
    "ticket",
    "satispay",
    "uber",
    "just_eat",
    "glovo",
    "deliveroo",
]


def get_features(df):
    return [c for c in df.columns if c not in EXCLUDE]


def walk_forward_backtest(
    input_file=MODEL_DATASET,
    output_file="data/processed/walk_forward_modular.csv",
):
    df = pd.read_csv(input_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    features = get_features(df)
    start_test = len(df) - TEST_DAYS

    results = pd.DataFrame()
    results["date"] = df.iloc[start_test:]["date"].values

    print("=" * 65)
    print("WALK FORWARD MODULARE")
    print("=" * 65)
    print("Feature:", len(features))
    print("Test days:", TEST_DAYS)

    for target in TARGETS:
        real = []
        pred = []

        for i in range(start_test, len(df)):
            train = df.iloc[:i]
            test = df.iloc[[i]]

            model = RandomForestRegressor(
                n_estimators=500,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                min_samples_leaf=2,
            )

            model.fit(train[features], train[target])

            p = model.predict(test[features])[0]

            pred.append(p)
            real.append(test[target].values[0])

        real = pd.Series(real)
        pred = pd.Series(pred)

        mae = mean_absolute_error(real, pred)
        mape = (
            (abs(real - pred) / real.replace(0, pd.NA))
            .dropna()
            .mean()
            * 100
        )

        print(f"\n{target.upper()}")
        print(f"MAE  : {mae:.2f} €")
        print(f"MAPE : {mape:.2f} %")

        results[f"{target}_real"] = real
        results[f"{target}_pred"] = pred
        results[f"{target}_error"] = pred - real

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results.to_csv(output_file, index=False)

    print("\nSalvato:", output_file)


if __name__ == "__main__":
    walk_forward_backtest()
