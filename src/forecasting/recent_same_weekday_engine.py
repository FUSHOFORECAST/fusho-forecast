import os
import pandas as pd

INPUT = "data/processed/feature_store_state.csv"

OUTPUT = "reports/recent_same_weekday"

os.makedirs(OUTPUT, exist_ok=True)


LOOKBACK = 8


def build_candidate(df):

    forecasts = []

    for _, row in df.iterrows():

        history = df[
            (df["date"] < row["date"])
            &
            (df["dayofweek"] == row["dayofweek"])
        ].tail(LOOKBACK)

        if len(history) == 0:

            pred = row["total_rolling_30"]

        else:

            weights = list(range(1, len(history)+1))

            pred = (
                history["total"] *
                weights
            ).sum() / sum(weights)

        forecasts.append(pred)

    return forecasts


def main():

    df = pd.read_csv(INPUT)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    df["recent_same_weekday_forecast"] = build_candidate(df)

    df["recent_same_weekday_error"] = (
        df["recent_same_weekday_forecast"] - df["total"]
    ).abs()

    df["recent_same_weekday_ape"] = (
        df["recent_same_weekday_error"] /
        df["total"] * 100
    )

    print("="*90)
    print("RECENT SAME WEEKDAY ENGINE")
    print("="*90)

    print()

    print(
        "MAPE:",
        round(
            df["recent_same_weekday_ape"].mean(),
            2
        )
    )

    print()

    print(df[
        [
            "date",
            "total",
            "recent_same_weekday_forecast",
            "recent_same_weekday_ape"
        ]
    ].tail(20).to_string(index=False))

    out = os.path.join(
        OUTPUT,
        "recent_same_weekday_candidate.csv"
    )

    df.to_csv(out, index=False)

    print()

    print("Salvato:")

    print(out)


if __name__ == "__main__":
    main()
