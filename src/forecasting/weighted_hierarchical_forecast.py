import os
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor

INPUT_FILE = "data/processed/model_dataset_share.csv"
OUTPUT_FILE = "data/processed/forecast_weighted_hierarchical_days.csv"

EXCLUDE = [
    "date", "source_file", "sheet", "holiday_name",
    "total", "delivery", "digital", "cash",
    "delivery_share", "digital_share", "cash_share",
    "delivery_share_target", "digital_share_target", "cash_share_target",
    "pos", "ticket", "satispay", "uber", "just_eat", "glovo", "deliveroo",
]


def make_weights(dates, last_date):
    dates = pd.to_datetime(dates)
    age_days = (last_date - dates).dt.days

    weights = pd.Series(0.35, index=dates.index)

    weights[age_days <= 365] = 0.60
    weights[age_days <= 180] = 0.80
    weights[age_days <= 90] = 0.95
    weights[age_days <= 30] = 1.20

    return weights


def run_forecast(days=14):
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    features = [c for c in df.columns if c not in EXCLUDE]

    history = df.copy()
    results = []

    for _ in range(days):
        last = history.iloc[-1]
        last_date = history["date"].max()
        future_date = last["date"] + timedelta(days=1)

        new_row = last.copy()
        new_row["date"] = future_date

        new_row["year"] = future_date.year
        new_row["month"] = future_date.month
        new_row["day"] = future_date.day
        new_row["dayofweek"] = future_date.dayofweek
        new_row["week"] = future_date.isocalendar().week
        new_row["dayofyear"] = future_date.dayofyear
        new_row["is_weekend"] = int(future_date.dayofweek in [5, 6])
        new_row["is_month_start"] = int(future_date.is_month_start)
        new_row["is_month_end"] = int(future_date.is_month_end)

        for col in ["total", "delivery", "digital", "cash"]:
            new_row[f"{col}_lag_1"] = history[col].iloc[-1]
            new_row[f"{col}_lag_7"] = history[col].iloc[-7]
            new_row[f"{col}_lag_14"] = history[col].iloc[-14]
            new_row[f"{col}_lag_21"] = history[col].iloc[-21]
            new_row[f"{col}_lag_28"] = history[col].iloc[-28]
            new_row[f"{col}_lag_30"] = history[col].iloc[-30]

            new_row[f"{col}_rolling_7"] = history[col].tail(7).mean()
            new_row[f"{col}_rolling_14"] = history[col].tail(14).mean()
            new_row[f"{col}_rolling_30"] = history[col].tail(30).mean()

        new_row["total_trend"] = new_row["total_rolling_7"] / new_row["total_rolling_30"]
        new_row["delivery_trend"] = new_row["delivery_rolling_7"] / new_row["delivery_rolling_30"]
        new_row["digital_trend"] = new_row["digital_rolling_7"] / new_row["digital_rolling_30"]
        new_row["cash_trend"] = new_row["cash_rolling_7"] / new_row["cash_rolling_30"]

        new_row["delivery_share_lag"] = new_row["delivery_lag_7"] / new_row["total_lag_7"]
        new_row["digital_share_lag"] = new_row["digital_lag_7"] / new_row["total_lag_7"]
        new_row["cash_share_lag"] = new_row["cash_lag_7"] / new_row["total_lag_7"]

        train = history.copy()

        x_train = train[features]
        x_test = pd.DataFrame([new_row])[features]

        weights = make_weights(train["date"], last_date)

        model = RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=3,
        )

        model.fit(x_train, train["total"], sample_weight=weights)
        total_pred = model.predict(x_test)[0]

        model.fit(x_train, train["delivery_share_target"], sample_weight=weights)
        delivery_share = model.predict(x_test)[0]

        model.fit(x_train, train["digital_share_target"], sample_weight=weights)
        digital_share = model.predict(x_test)[0]

        model.fit(x_train, train["cash_share_target"], sample_weight=weights)
        cash_share = model.predict(x_test)[0]

        delivery_share = max(0, delivery_share)
        digital_share = max(0, digital_share)
        cash_share = max(0, cash_share)

        share_sum = delivery_share + digital_share + cash_share

        if share_sum == 0:
            delivery_share = 0.68
            digital_share = 0.22
            cash_share = 0.10
            share_sum = 1

        delivery_share /= share_sum
        digital_share /= share_sum
        cash_share /= share_sum

        delivery_pred = total_pred * delivery_share
        digital_pred = total_pred * digital_share
        cash_pred = total_pred * cash_share

        results.append({
            "date": future_date,
            "total_pred": round(total_pred, 2),
            "delivery_pred": round(delivery_pred, 2),
            "digital_pred": round(digital_pred, 2),
            "cash_pred": round(cash_pred, 2),
            "delivery_share": round(delivery_share, 4),
            "digital_share": round(digital_share, 4),
            "cash_share": round(cash_share, 4),
        })

        new_row["total"] = total_pred
        new_row["delivery"] = delivery_pred
        new_row["digital"] = digital_pred
        new_row["cash"] = cash_pred
        new_row["delivery_share_target"] = delivery_share
        new_row["digital_share_target"] = digital_share
        new_row["cash_share_target"] = cash_share

        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    forecast = pd.DataFrame(results)

    os.makedirs("data/processed", exist_ok=True)
    forecast.to_csv(OUTPUT_FILE, index=False)

    print("\nFORECAST PESATO GERARCHICO")
    print(forecast)

    print("\nTOTALE:", round(forecast["total_pred"].sum(), 2))
    print("DELIVERY:", round(forecast["delivery_pred"].sum(), 2))
    print("DIGITAL:", round(forecast["digital_pred"].sum(), 2))
    print("CASH:", round(forecast["cash_pred"].sum(), 2))
    print("\nSalvato:", OUTPUT_FILE)


if __name__ == "__main__":
    run_forecast(days=14)
