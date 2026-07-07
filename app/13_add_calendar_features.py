import os
from datetime import timedelta

import holidays
import pandas as pd

INPUT_FILE = "data/processed/master_dataset_weather.csv"
OUTPUT_FILE = "data/processed/master_dataset_weather_calendar.csv"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

it_holidays = holidays.Italy(years=range(2023, 2027))

holiday_dates = set(it_holidays.keys())

df["is_holiday"] = df["date"].dt.date.apply(
    lambda d: int(d in holiday_dates)
)

df["holiday_name"] = df["date"].dt.date.apply(
    lambda d: it_holidays.get(d, "")
)

df["dayofweek"] = df["date"].dt.dayofweek

df["is_pre_holiday"] = df["date"].dt.date.apply(
    lambda d: int((d + timedelta(days=1)) in holiday_dates)
)

df["is_post_holiday"] = df["date"].dt.date.apply(
    lambda d: int((d - timedelta(days=1)) in holiday_dates)
)

def is_bridge(row):
    d = row["date"].date()
    wd = row["dayofweek"]

    # lunedì prima di un martedì festivo
    if wd == 0 and (d + timedelta(days=1)) in holiday_dates:
        return 1

    # venerdì dopo un giovedì festivo
    if wd == 4 and (d - timedelta(days=1)) in holiday_dates:
        return 1

    return 0

df["is_bridge"] = df.apply(is_bridge, axis=1)

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("CALENDAR FEATURES CREATE")
print("=" * 60)

print(df[
    [
        "date",
        "holiday_name",
        "is_holiday",
        "is_pre_holiday",
        "is_post_holiday",
        "is_bridge",
    ]
].tail(30))

print("\nSalvato:")
print(OUTPUT_FILE)
