import pandas as pd
import joblib
from datetime import timedelta

DATA = "data/processed/model_dataset.csv"

FEATURES = [
    "dayofweek",
    "day",
    "month",
    "year",
    "dayofyear",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "total_lag_7",
    "total_rolling_7",
    "total_rolling_30",
    "delivery_lag_7",
    "delivery_rolling_7",
    "digital_lag_7",
    "digital_rolling_7",
    "cash_lag_7",
    "cash_rolling_7",
]

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

last_row = df.iloc[-1]
last_date = last_row["date"]

future_rows = []

history = df.copy()

for i in range(1, 8):
    date = last_date + timedelta(days=i)

    row = {
        "date": date,
        "dayofweek": date.dayofweek,
        "day": date.day,
        "month": date.month,
        "year": date.year,
        "dayofyear": date.dayofyear,
        "is_weekend": int(date.dayofweek in [5, 6]),
        "is_month_start": int(date.is_month_start),
        "is_month_end": int(date.is_month_end),
        "total_lag_7": history["total"].iloc[-7],
        "total_rolling_7": history["total"].tail(7).mean(),
        "total_rolling_30": history["total"].tail(30).mean(),
        "delivery_lag_7": history["delivery"].iloc[-7],
        "delivery_rolling_7": history["delivery"].tail(7).mean(),
        "digital_lag_7": history["digital"].iloc[-7],
        "digital_rolling_7": history["digital"].tail(7).mean(),
        "cash_lag_7": history["cash"].iloc[-7],
        "cash_rolling_7": history["cash"].tail(7).mean(),
    }

    X = pd.DataFrame([row])[FEATURES]

    total_model = joblib.load("models/total_model.pkl")
    delivery_model = joblib.load("models/delivery_model.pkl")
    digital_model = joblib.load("models/digital_model.pkl")
    cash_model = joblib.load("models/cash_model.pkl")

    row["total_pred"] = total_model.predict(X)[0]
    row["delivery_pred"] = delivery_model.predict(X)[0]
    row["digital_pred"] = digital_model.predict(X)[0]
    row["cash_pred"] = cash_model.predict(X)[0]

    # totale coerente con la somma dei reparti
    row["total_by_parts"] = row["delivery_pred"] + row["digital_pred"] + row["cash_pred"]

    future_rows.append(row)

    # aggiunge la previsione alla history per calcolare i giorni successivi
    new_history_row = {
        **row,
        "total": row["total_by_parts"],
        "delivery": row["delivery_pred"],
        "digital": row["digital_pred"],
        "cash": row["cash_pred"],
    }

    history = pd.concat([history, pd.DataFrame([new_history_row])], ignore_index=True)

forecast = pd.DataFrame(future_rows)

output = forecast[[
    "date",
    "total_by_parts",
    "delivery_pred",
    "digital_pred",
    "cash_pred",
]]

output.to_csv("data/processed/forecast_7_days.csv", index=False)

print("\nFORECAST 7 GIORNI")
print(output)

print("\nTOTALE SETTIMANA PREVISTO:", round(output["total_by_parts"].sum(), 2))
print("DELIVERY SETTIMANA:", round(output["delivery_pred"].sum(), 2))
print("DIGITAL SETTIMANA:", round(output["digital_pred"].sum(), 2))
print("CASH SETTIMANA:", round(output["cash_pred"].sum(), 2))
