import os
from datetime import timedelta

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

INPUT_FILE = "data/processed/model_dataset_share.csv"
OUTPUT_FILE = "data/processed/forecast_adaptive_memory_v2.csv"

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

    "delivery_share_target",
    "digital_share_target",
    "cash_share_target",

    "pos",
    "ticket",
    "satispay",
    "uber",
    "just_eat",
    "glovo",
    "deliveroo",
]


def create_future_row(history, future_date):
    last = history.iloc[-1].copy()
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

    return new_row


def filter_recent(df, last_date, days):
    start_date = last_date - pd.Timedelta(days=days)
    return df[df["date"] >= start_date].copy()


def run_forecast(days=14):
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    features = [c for c in df.columns if c not in EXCLUDE]

    history = df.copy()
    results = []

    for _ in range(days):
        last_date = history["date"].max()
        future_date = last_date + timedelta(days=1)

        new_row = create_future_row(history, future_date)
        x_test = pd.DataFrame([new_row])[features]

        # MEMORIA 1: incasso recente
        # Usa solo ultimo anno, non tutto lo storico.
        revenue_train = filter_recent(history, last_date, 365)

        # MEMORIA 2: modello breve per trend recente
        # Usa ultimi 120 giorni.
        short_train = filter_recent(history, last_date, 120)

        # MEMORIA 3: stagionalità storica
        # Cerca stesso mese + stesso giorno settimana nello storico.
        seasonal_train = history[
            (history["month"] == future_date.month)
            & (history["dayofweek"] == future_date.dayofweek)
        ].copy()

        if len(seasonal_train) < 5:
            seasonal_train = history[history["month"] == future_date.month].copy()

        model_revenue = RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=3,
        )

        model_short = RandomForestRegressor(
            n_estimators=300,
            random_state=43,
            n_jobs=-1,
            min_samples_leaf=2,
        )

        model_revenue.fit(revenue_train[features], revenue_train["total"])
        pred_revenue = model_revenue.predict(x_test)[0]

        model_short.fit(short_train[features], short_train["total"])
        pred_short = model_short.predict(x_test)[0]

        if len(seasonal_train) >= 3:
            pred_seasonal = seasonal_train["total"].median()
        else:
            pred_seasonal = history["total"].tail(365).median()

        # Combina le tre memorie
        # Incasso recente conta di più.
        total_pred = (
            pred_revenue * 0.55
            + pred_short * 0.30
            + pred_seasonal * 0.15
        )

        # Mix canali: molto recente
        share_recent = history.tail(90)

        delivery_share = share_recent["delivery_share_target"].mean()
        digital_share = share_recent["digital_share_target"].mean()
        cash_share = share_recent["cash_share_target"].mean()

        share_sum = delivery_share + digital_share + cash_share

        delivery_share /= share_sum
        digital_share /= share_sum
        cash_share /= share_sum

        delivery_pred = total_pred * delivery_share
        digital_pred = total_pred * digital_share
        cash_pred = total_pred * cash_share

        results.append({
            "date": future_date,
            "pred_revenue_memory": round(pred_revenue, 2),
            "pred_short_memory": round(pred_short, 2),
            "pred_seasonal_memory": round(pred_seasonal, 2),
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

    print("=" * 80)
    print("ADAPTIVE MEMORY FORECAST V2")
    print("=" * 80)
    print(forecast.to_string(index=False))

    print("\nTOTALE:", round(forecast["total_pred"].sum(), 2))
    print("DELIVERY:", round(forecast["delivery_pred"].sum(), 2))
    print("DIGITAL:", round(forecast["digital_pred"].sum(), 2))
    print("CASH:", round(forecast["cash_pred"].sum(), 2))
    print("\nSalvato:", OUTPUT_FILE)


if __name__ == "__main__":
    run_forecast(days=14)
